use std::{
    iter::Sum,
    ops::{AddAssign, Neg, Sub},
};

use num_traits::Zero;

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn sub<J, RhsG>(
        &self,
        rhs: &Tensor<F, J, RhsG>,
    ) -> Result<Tensor<F, Contiguous, Grad>, PzeudoErr>
    where
        F: Copy + Sub<Output = F> + Zero + Clone,
        for<'a> ArrayRef<'a, F, T>: OpsSub<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsSub<F> + OpsBroadcast<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> =
            storage.get_as_array_ref(self.get_array_idx(), ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, F, J> =
            storage.get_as_array_ref(rhs.get_array_idx(), ContiguousType::Arr)?;

        let array = OpsSub::sub(&lhs_array, &rhs_array)?;
        let shape = array.shape.to_vec();
        let (lhs_broadcast, rhs_broadcast) = broadcast_detect(lhs_array.shape, rhs_array.shape);

        let grad = Array::<F>::zeros(&shape);
        let array_idx = storage.push(ElementType::Arr(array))?;
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);

        let record_label = RecordLabel::Sub(
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

pub fn sub_backward<F>(
    gradient_idx: Option<StorageType>,
    lhs_grad: Option<StorageType>,
    lhs_broadcast_dim: Option<&Vec<usize>>,
    rhs_grad: Option<StorageType>,
    rhs_broadcast_dim: Option<&Vec<usize>>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Clone + AddAssign + Copy + Neg<Output = F> + Sum<&'a F> + Clone + Zero,
{
    // f(lhs, rhs) = lhs - rhs
    if let Some(gradient_idx) = gradient_idx {
        let gradient = storage
            .get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?
            .into_array();

        if let Some(lhs_grad) = lhs_grad {
            // df(lhs, rhs)/dlhs = 1 * gradient
            let mut lhs_gradient =
                storage.get_as_array_ref_mut::<View>(lhs_grad, ContiguousType::Grad)?;
            match lhs_broadcast_dim {
                Some(dim) => {
                    let grad = gradient.sum_axis(dim, true)?;
                    let to_shape = grad.to_shape(lhs_gradient.shape)?;
                    lhs_gradient.add_assign(&to_shape)?
                }
                None => lhs_gradient.add_assign(&gradient)?,
            };
        }

        if let Some(rhs_grad) = rhs_grad {
            // df(lhs, rhs)/drhs = -1 * gradient
            let grad = gradient.neg()?;

            let mut rhs_gradient =
                storage.get_as_array_ref_mut::<View>(rhs_grad, ContiguousType::Grad)?;
            match rhs_broadcast_dim {
                Some(dim) => {
                    let grad = grad.sum_axis(dim, true)?;
                    let to_shape = grad.to_shape(rhs_gradient.shape)?;
                    rhs_gradient.add_assign(&to_shape)?
                }
                None => rhs_gradient.add_assign(&grad)?,
            };
        }
    }
    Ok(())
}
