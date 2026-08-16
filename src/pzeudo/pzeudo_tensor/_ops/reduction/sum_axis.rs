use std::{marker::PhantomData, ops::AddAssign};

use num_traits::{Float, Zero};

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float + AddAssign,
{
    pub fn sum_axis<ReqGrad>(
        &self,
        axis: &[usize],
        keep_dim: bool,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let sum = OpsSum::sum_axis(&array, axis, keep_dim)?;
        let shape = sum.shape.to_vec();
        let array_idx = storage.push(ElementType::Arr(sum))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::SumAxis(self.grad_idx, axis.to_vec(), keep_dim, grad_idx);
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

pub fn sum_axis_backward<F>(
    array_grad_idx: Option<StorageType>,
    axis: &Vec<usize>,
    keep_dim: bool,
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

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;

            if !keep_dim {
                let shape = grad_ref.shape;
                let len = axis.len() + shape.len();
                let mut broadcast = vec![0; len];
                for dim in axis {
                    broadcast[*dim] = 1;
                }

                let mut idx = 0;
                for dim in &mut broadcast {
                    if *dim == 0 {
                        *dim = shape[idx];
                        idx += 1;
                    }
                }

                let grad_ref = ArrayRef::<F, Contiguous> {
                    data: grad_ref.data,
                    offset: grad_ref.offset,
                    shape: &broadcast,
                    stride: &shape_to_stride(&broadcast),
                    _array_type: PhantomData::default(),
                };

                let broadcasted = grad_ref.broadcast(array_grad.shape)?;
                array_grad.add_assign(&broadcasted)?;
            } else {
                let broadcasted = grad_ref.broadcast(array_grad.shape)?;
                array_grad.add_assign(&broadcasted)?;
            };

            storage.replace_grad(grad_idx, grad)?;
        }
    }

    Ok(())
}
