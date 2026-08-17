use std::ops::{AddAssign, Div};

use num_traits::{Float, NumCast, Zero};

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float + AddAssign,
{
    pub fn avg<ReqGrad>(
        &self,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let avg = array.avg()?;
        let array_idx = storage.push(ElementType::Arr(avg))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&[1], &mut storage)?;

        let record_label = RecordLabel::Avg(self.grad_idx, grad_idx);
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

pub fn avg_backward<F>(
    array_grad_idx: Option<StorageType>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + AddAssign + NumCast + Div<Output = F>,
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

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;

            let len = array_grad.shape.iter().product::<usize>();
            let n = F::from(len).ok_or(PzeudoErr::BackwardErr(format!(
                "avg_backward. Cannot perform data type casting on length."
            )))?;

            let grad_val = grad_ref.linear_index(0)? / n;
            for i in 0..len {
                *array_grad.linear_index_mut(i)? += grad_val;
            }

            storage.replace_grad(grad_idx, grad)?;
        }
    }

    Ok(())
}
