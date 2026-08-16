use crate::prelude::*;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float + AddAssign,
{
    pub fn sum<ReqGrad>(
        &self,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let sum = array.sum()?;
        let array_idx = storage.push(ElementType::Arr(sum))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&[1], &mut storage)?;

        let record_label = RecordLabel::Sum(self.grad_idx, grad_idx);
        let mut record = self.record.borrow_mut();
        let record_idx = record.len();
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            vec![1],
            Some(RecordStatus::Record(record_idx)),
            self.record.clone(),
            self.storage.clone(),
        );

        Ok(tensor)
    }
}

pub fn sum_backward<F>(
    array_grad_idx: Option<StorageType>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + AddAssign,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        }

        if let Some(array_grad_idx) = array_grad_idx {
            storage.set_grad_update(array_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(array_grad_idx, storage)? {
                return Ok(());
            }

            let grad = storage.take_grad(grad_idx)?;
            let grad_ref = grad.to_array_ref::<Contiguous>();
            let grad_val = grad_ref.linear_index(0)?;

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;

            let len = array_grad.shape.iter().product::<usize>();
            for i in 0..len {
                *array_grad.linear_index_mut(i)? += grad_val;
            }

            storage.replace_grad(grad_idx, grad)?;
        }
    }

    Ok(())
}
