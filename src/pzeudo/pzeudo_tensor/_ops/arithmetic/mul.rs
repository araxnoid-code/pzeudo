use std::{
    iter::Sum,
    ops::{AddAssign, Mul},
};

use num_traits::Zero;

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn mul<J, RhsGrad, OutGrad>(
        &self,
        rhs: &Tensor<F, J, RhsGrad>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
        F: Copy + Mul<Output = F> + Zero + Clone,
        for<'a> ArrayRef<'a, F, T>: OpsMul<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsMul<F> + OpsBroadcast<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> =
            storage.get_as_array_ref(self.get_array_idx(), ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, F, J> =
            storage.get_as_array_ref(rhs.get_array_idx(), ContiguousType::Arr)?;

        let array = OpsMul::mul(&lhs_array, &rhs_array)?;
        let shape = array.shape.to_vec();
        let (lhs_broadcast, rhs_broadcast) = broadcast_detect(lhs_array.shape, rhs_array.shape);

        let array_idx = storage.push(ElementType::Arr(array))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Mul(
            (self.get_array_idx(), self.get_grad_idx(), lhs_broadcast),
            (rhs.get_array_idx(), rhs.get_grad_idx(), rhs_broadcast),
            grad_idx,
        );

        let mut record = self.get_record().borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.record.clone(),
            self.storage.clone(),
        ))
    }
}

pub fn mul_backward<F>(
    gradient_idx: Option<StorageType>,
    lhs: StorageType,
    rhs: StorageType,
    lhs_grad: Option<StorageType>,
    lhs_broadcast_dim: Option<&Vec<usize>>,
    rhs_grad: Option<StorageType>,
    rhs_broadcast_dim: Option<&Vec<usize>>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Clone + AddAssign + Copy + Mul<Output = F> + Sum<&'a F> + Zero,
    for<'a> ArrayRef<'a, F, View>: OpsMul<F> + OpsBroadcast<F>,
{
    // f(lhs, rhs) = lhs * rhs
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();

        if let Some(lhs_grad) = lhs_grad {
            storage.set_grad_update(lhs_grad, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad, storage)? {
                // df(lhs, rhs)/dlhs = rhs * gradient
                let mut lhs_gradient = storage.take_grad(lhs_grad)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                let rhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(rhs, ContiguousType::Arr)?;

                match lhs_broadcast_dim {
                    Some(dim) => {
                        let grad = rhs_value.mul(&gradient_ref)?;
                        let gradient = grad.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(lhs_gradient_ref.shape)?;
                        lhs_gradient_ref.add_assign(&to_shape)?
                    }
                    None => {
                        let len = rhs_value.shape.iter().product::<usize>();
                        for i in 0..len {
                            *lhs_gradient_ref.linear_index_mut(i)? +=
                                rhs_value.linear_index(i)? * gradient_ref.linear_index(i)?;
                        }
                    }
                }

                storage.replace_grad(lhs_grad, lhs_gradient)?;
            };
        }

        if let Some(rhs_grad) = rhs_grad {
            storage.set_grad_update(rhs_grad, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(rhs_grad, storage)? {
                // df(lhs, rhs)/drhs = lhs * gradient
                let mut rhs_gradient = storage.take_grad(rhs_grad)?;
                let mut rhs_gradient_ref = rhs_gradient.to_array_ref_mut::<View>();

                let lhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(lhs, ContiguousType::Arr)?;

                match rhs_broadcast_dim {
                    Some(dim) => {
                        let grad = lhs_value.mul(&gradient_ref)?;
                        let gradient = grad.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(rhs_gradient_ref.shape)?;
                        rhs_gradient_ref.add_assign(&to_shape)?
                    }
                    None => {
                        let len = lhs_value.shape.iter().product::<usize>();
                        for i in 0..len {
                            *rhs_gradient_ref.linear_index_mut(i)? +=
                                lhs_value.linear_index(i)? * gradient_ref.linear_index(i)?;
                        }
                    }
                };

                storage.replace_grad(rhs_grad, rhs_gradient)?;
            }
        }

        storage.replace_grad(gradient_idx, gradient)?;
    }

    Ok(())
}
