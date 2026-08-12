pub use crate::prelude::*;
use num_traits::{Float, One, Zero};
use std::ops::AddAssign;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float,
{
    /// f(x) = √x
    pub fn sqrt<OutGrad>(
        &self,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.sqrt()?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Sqrt((array_idx, self.get_grad_idx()), grad_idx);
        self.get_record().borrow_mut().push(Some(record_label));

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }
}

/// - f(x) = √x
/// - df(x)/x = 1/2√x * gradient = gradient/2√x
pub fn sqrt_backward<F>(
    out_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: One + Float + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                // - f(x) = √x
                // - df(x)/x = 1/2√x * gradient = gradient/2√x
                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();
                let out_value =
                    storage.get_as_array_ref::<Contiguous>(out_idx, ContiguousType::Arr)?;

                let len = out_value.shape.iter().product::<usize>();
                let two = F::one() + F::one();
                for i in 0..len {
                    *lhs_gradient_ref.linear_index_mut(i)? +=
                        gradient_ref.linear_index(i)? / (two * out_value.linear_index(i)?);
                }
                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
            }
        }
        storage.replace_grad(gradient_idx, gradient)?;
    }
    Ok(())
}
