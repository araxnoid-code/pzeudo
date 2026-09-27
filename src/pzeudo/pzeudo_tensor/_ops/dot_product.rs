use std::ops::{AddAssign, Mul};

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn dot<RhsT, RhsG, ReqGrad>(
        &self,
        rhs: &Tensor<F, RhsT, RhsG>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F> + DotProductF32F64<F, T>,
        for<'a> ArrayRef<'a, F, RhsT>: ArrayTrait<F>,
    {
        let mut storage = self.get_storage().borrow_mut();
        let lhs_array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let rhs_array =
            storage.get_as_array_ref::<RhsT>(rhs.get_array_idx(), ContiguousType::Arr)?;
        let dot = lhs_array.dot_f32f64(&rhs_array)?;
        let shape = vec![1];

        let array_idx = storage.push(ElementType::Arr(dot))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_idx = if requires_grad.is_grad() {
            let mut record = self.get_record().borrow_mut();
            let record_idx = RecordStatus::Record(record.len());
            let label = RecordLabel::DotProduct(
                self.get_array_idx(),
                self.get_grad_idx(),
                rhs.get_array_idx(),
                rhs.get_grad_idx(),
                grad_idx,
            );
            record.push(label);
            Some(record_idx)
        } else {
            None
        };

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_idx,
            self.get_record().clone(),
            self.get_storage().clone(),
        );
        Ok(tensor)
    }
}

pub fn dot_backward<F>(
    lhs_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    rhs_idx: StorageType,
    rhs_grad_idx: Option<StorageType>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + AddAssign + Mul<Output = F>,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        let grad = storage.get_as_array_ref::<Contiguous>(grad_idx, ContiguousType::Grad)?;
        let grad_val = grad.data[0];

        // PATH 1
        if let (Some(lhs_grad_idx), Some(rhs_grad_idx)) = (lhs_grad_idx, rhs_grad_idx) {
            storage.set_grad_update(lhs_grad_idx, true)?;
            storage.set_grad_update(rhs_grad_idx, true)?;

            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)?
                && !is_no_grad_or_time_not_match_or_no_update(rhs_grad_idx, storage)?
            {
                // GRADIENT
                let mut lhs_grad_take = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_grad = lhs_grad_take.to_array_ref_mut::<View>();

                let mut rhs_grad_take = storage.take_grad(rhs_grad_idx)?;
                let mut rhs_grad = rhs_grad_take.to_array_ref_mut::<View>();

                // ARRAY
                let lhs_array = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let rhs_array = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;

                let len = lhs_array.shape.iter().product::<usize>();
                for i in 0..len {
                    *lhs_grad.linear_index_mut(i)? += rhs_array.linear_index(i)? * grad_val;
                    *rhs_grad.linear_index_mut(i)? += lhs_array.linear_index(i)? * grad_val;
                }

                storage.replace_grad(lhs_grad_idx, lhs_grad_take)?;
                storage.replace_grad(rhs_grad_idx, rhs_grad_take)?;
                return Ok(());
            }
        }

        // PATH 2
        if let Some(lhs_grad_idx) = lhs_grad_idx {
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                // GRADIENT
                let mut lhs_grad_take = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_grad = lhs_grad_take.to_array_ref_mut::<View>();

                // ARRAY
                let rhs_array = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;

                let len = rhs_array.shape.iter().product::<usize>();
                for i in 0..len {
                    *lhs_grad.linear_index_mut(i)? += rhs_array.linear_index(i)? * grad_val;
                }

                storage.replace_grad(lhs_grad_idx, lhs_grad_take)?;
            }
        }

        if let Some(rhs_grad_idx) = rhs_grad_idx {
            storage.set_grad_update(rhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(rhs_grad_idx, storage)? {
                // GRADIENT
                let mut rhs_grad_take = storage.take_grad(rhs_grad_idx)?;
                let mut rhs_grad = rhs_grad_take.to_array_ref_mut::<View>();

                // ARRAY
                let lhs_array = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;

                let len = lhs_array.shape.iter().product::<usize>();
                for i in 0..len {
                    *rhs_grad.linear_index_mut(i)? += lhs_array.linear_index(i)? * grad_val;
                }

                storage.replace_grad(rhs_grad_idx, rhs_grad_take)?;
            }
        }
    }

    Ok(())
}
