use std::ops::AddAssign;

use crate::prelude::*;
use num_traits::Float;
impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Float,
{
    /// f(x) = tan(x)
    pub fn tan<OutGrad>(
        &self,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.tan()?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Tan((array_idx, self.get_grad_idx()), grad_idx);
        let mut record = self.get_record().borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }
}

/// - f(x) = tan(x)
/// - df(x)/x = (tan(x)^2 + 1) * gradient
pub fn tan_backward<F>(
    out_idx: StorageType,
    array_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        if let Some(lhs_grad_idx) = array_grad_idx {
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                let gradient = storage.take_grad(gradient_idx)?;
                let gradient_ref = gradient.to_array_ref::<Contiguous>();

                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();
                let out_value =
                    storage.get_as_array_ref::<Contiguous>(out_idx, ContiguousType::Arr)?;

                let len = out_value.shape.iter().product::<usize>();
                let one = F::one();
                for i in 0..len {
                    let out_val = out_value.linear_index(i)?;
                    *lhs_gradient_ref.linear_index_mut(i)? +=
                        (out_val * out_val + one) * gradient_ref.linear_index(i)?;
                }
                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
                storage.replace_grad(gradient_idx, gradient)?;
            }
        }
    }
    Ok(())
}
