use std::{
    iter::Sum,
    marker::PhantomData,
    ops::{AddAssign, Div, Neg},
};

use num_traits::{Float, One, Zero};

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn div<J, RhsGrad, OutGrad>(
        &self,
        rhs: &Tensor<F, J, RhsGrad>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
        F: Copy + Div<Output = F> + Zero + Clone,
        for<'a> ArrayRef<'a, F, T>: OpsDiv<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsDiv<F> + OpsBroadcast<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> =
            storage.get_as_array_ref(self.get_array_idx(), ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, F, J> =
            storage.get_as_array_ref(rhs.get_array_idx(), ContiguousType::Arr)?;

        let array = OpsDiv::div(&lhs_array, &rhs_array)?;
        let shape = array.shape.to_vec();
        let (lhs_broadcast, rhs_broadcast) = broadcast_detect(lhs_array.shape, rhs_array.shape);

        let array_idx = storage.push(ElementType::Arr(array))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Div(
            (self.get_array_idx(), self.get_grad_idx(), lhs_broadcast),
            (rhs.get_array_idx(), rhs.get_grad_idx(), rhs_broadcast),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        ))
    }
}

pub fn div_backward<F>(
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
    for<'a> F: Clone
        + AddAssign
        + Copy
        + Div<Output = F>
        + One
        + Neg<Output = F>
        + Float
        + Zero
        + Clone
        + Sum<&'a F>,
    for<'a> ArrayRef<'a, F, Contiguous>: OpsBroadcast<F>,
{
    // f(lhs, rhs) = lhs / rhs
    if let Some(gradient_idx) = gradient_idx {
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        };
        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();

        if let Some(lhs_grad) = lhs_grad {
            if !check_no_grad_or_time_not_match(lhs_grad, storage)? {
                let mut lhs_gradient = storage.take_grad(lhs_grad)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                // df(lhs, rhs)/dlhs = 1/rhs * gradient = gradient / rhs
                let rhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(rhs, ContiguousType::Arr)?;

                match lhs_broadcast_dim {
                    Some(dim) => {
                        let grad = OpsDiv::div(&gradient_ref, &rhs_value)?;
                        let gradient = grad.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(lhs_gradient_ref.shape)?;
                        lhs_gradient_ref.add_assign(&to_shape)?
                    }
                    None => {
                        let len = lhs_gradient_ref.shape.iter().product::<usize>();
                        for i in 0..len {
                            *lhs_gradient_ref.linear_index_mut(i)? +=
                                gradient_ref.linear_index(i)? / rhs_value.linear_index(i)?;
                        }
                    }
                }

                storage.replace_grad(lhs_grad, lhs_gradient)?;
            }
        }

        if let Some(rhs_grad) = rhs_grad {
            if !check_no_grad_or_time_not_match(rhs_grad, storage)? {
                // df(lhs, rhs)/drhs = -lhs/rhs^2 * gradient
                let mut rhs_gradient = storage.take_grad(rhs_grad)?;
                let mut rhs_gradient_ref = rhs_gradient.to_array_ref_mut::<View>();

                let rhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(rhs, ContiguousType::Arr)?;
                let lhs_value: ArrayRef<'_, F, View> =
                    storage.get_as_array_ref(lhs, ContiguousType::Arr)?;

                match rhs_broadcast_dim {
                    Some(dim) => {
                        let rhs_broadcaseted = rhs_value.broadcast(lhs_value.shape)?;

                        let len = lhs_value.shape.iter().product::<usize>();
                        let mut grad_vec = Vec::with_capacity(len);
                        for i in 0..len {
                            let x = -lhs_value.linear_index(i)?
                                / rhs_broadcaseted.linear_index(i)?.powi(2)
                                * gradient_ref.linear_index(i)?;
                            grad_vec.push(x);
                        }

                        let grad: ArrayRef<'_, F, Contiguous> = ArrayRef {
                            data: &grad_vec,
                            offset: 0,
                            shape: lhs_value.shape,
                            stride: lhs_value.stride,
                            _array_type: PhantomData::default(),
                        };

                        let gradient = grad.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(rhs_gradient_ref.shape)?;
                        rhs_gradient_ref.add_assign(&to_shape)?
                    }
                    None => {
                        let len = lhs_value.shape.iter().product::<usize>();
                        for i in 0..len {
                            let x = -lhs_value.linear_index(i)?
                                / rhs_value.linear_index(i)?.powi(2)
                                * gradient_ref.linear_index(i)?;
                            *rhs_gradient_ref.linear_index_mut(i)? += x;
                        }
                        // rhs_gradient.add_assign(&grad)?
                    }
                }
                storage.replace_grad(rhs_grad, rhs_gradient)?;
            }
        }

        storage.replace_grad(gradient_idx, gradient)?;
    }
    Ok(())
}
