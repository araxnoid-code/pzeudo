use std::ops::AddAssign;

use num_traits::Float;

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Copy + Float,
{
    pub fn max_axis<ReqGrad>(
        &self,
        axis: &[usize],
        keep_dim: bool,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let (array_idx, gradient_idx, record_idx) = if requires_grad.is_grad() {
            let (array_max, array_argamax) = array.max_and_argmax_axis(axis, keep_dim)?;

            let array_idx = storage.push(ElementType::Arr(array_max))?;
            let gradient_idx = requires_grad.into_zeros_grad_storage(&[1], &mut storage)?;

            let mut record = self.get_record().borrow_mut();
            let record_idx = RecordStatus::Record(record.len());
            let record_label =
                RecordLabel::MaxAxis(self.grad_idx, array_argamax.data, gradient_idx);
            record.push(record_label);

            (array_idx, gradient_idx, Some(record_idx))
        } else {
            let array_max = array.max()?;

            let array_idx = storage.push(ElementType::Arr(array_max))?;

            (array_idx, None, None)
        };

        let tensor = Tensor::_new(
            array_idx,
            gradient_idx,
            vec![1],
            record_idx,
            self.get_record().clone(),
            self.get_storage().clone(),
        );

        Ok(tensor)
    }
}

pub fn max_axis_backward<F>(
    array_grad_idx: Option<StorageType>,
    indexs: &[F],
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + AddAssign + FToUsize,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, &storage)? {
            return Ok(());
        };

        if let Some(array_grad_idx) = array_grad_idx {
            storage.set_grad_update(array_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(array_grad_idx, &storage)? {
                return Ok(());
            };

            let grad_take = storage.take_grad(grad_idx)?;
            let grad = grad_take.to_array_ref::<Contiguous>();

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;
            for (i, f) in indexs.iter().enumerate() {
                let index = f.into_usize();
                *array_grad.linear_index_mut(index)? += grad.linear_index(i)?;
            }

            storage.replace_grad(grad_idx, grad_take)?;
        }
    }
    Ok(())
}
