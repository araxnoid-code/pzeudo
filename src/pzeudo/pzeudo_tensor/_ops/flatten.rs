use crate::prelude::*;
use num_traits::{Float, NumCast};
use std::ops::{AddAssign, Div};
impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Float,
{
    pub fn flatten<ReqGrad>(
        &self,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let to_shape = array.shape.to_vec();

        let flatten = array.flatten()?;
        let shape = flatten.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(flatten))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Flatten(self.grad_idx, to_shape, grad_idx);
        let mut record = self.record.borrow_mut();
        let record_idx = record.len();
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            Some(RecordStatus::Record(record_idx)),
            self.record.clone(),
            self.storage.clone(),
        );

        Ok(tensor)
    }
}

pub fn flatten_backward<F>(
    array_grad_idx: Option<StorageType>,
    to_shape: &[usize],
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

            let to_shape_grad = grad_ref.to_shape(to_shape)?;
            array_grad.add_assign(&to_shape_grad);

            storage.replace_grad(grad_idx, grad)?;
        }
    }

    Ok(())
}
