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
        let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;

        let record_label = RecordLabel::Mul(
            (self.get_array_idx(), self.get_grad_idx(), lhs_broadcast),
            (rhs.get_array_idx(), rhs.get_grad_idx(), rhs_broadcast),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
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
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad) = lhs_grad {
            if !check_no_grad_or_time_not_match(lhs_grad, storage)? {
                // df(lhs, rhs)/dlhs = rhs * gradient
                let rhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(rhs, ContiguousType::Arr)?;
                let grad = rhs_value.mul(&gradient)?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_grad, ContiguousType::Grad)?;
                match lhs_broadcast_dim {
                    Some(dim) => {
                        let gradient = grad.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(lhs_gradient.shape)?;
                        lhs_gradient.add_assign(&to_shape)?
                    }
                    None => lhs_gradient.add_assign(&grad)?,
                }
            };
        }

        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;
        if let Some(rhs_grad) = rhs_grad {
            if !check_no_grad_or_time_not_match(rhs_grad, storage)? {
                // df(lhs, rhs)/drhs = lhs * gradient
                let lhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(lhs, ContiguousType::Arr)?;
                let grad = lhs_value.mul(&gradient)?;

                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_grad, ContiguousType::Grad)?;
                match rhs_broadcast_dim {
                    Some(dim) => {
                        let gradient = grad.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(rhs_gradient.shape)?;
                        rhs_gradient.add_assign(&to_shape)?
                    }
                    None => rhs_gradient.add_assign(&grad)?,
                };
            }
        }
    }
    Ok(())
}
