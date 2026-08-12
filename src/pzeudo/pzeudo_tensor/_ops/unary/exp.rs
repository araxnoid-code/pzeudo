pub use crate::prelude::*;
use num_traits::{Float, One, Zero};
use std::ops::AddAssign;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float,
{
    pub fn exp<OutGrad>(
        &self,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.exp()?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Exp((array_idx, self.get_grad_idx()), grad_idx);

        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

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

pub fn exp_backward<F>(
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

        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                let out_value =
                    storage.get_as_array_ref::<Contiguous>(out_idx, ContiguousType::Arr)?;
                // e^x * gradient
                let grad = out_value.mul(&gradient)?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&grad)?;
            }
        }
    }
    Ok(())
}
