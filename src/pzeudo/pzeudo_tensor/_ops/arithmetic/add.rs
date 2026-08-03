use crate::prelude::*;
use num_traits::Zero;
use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

impl<F, T, G> Tensor<F, T, G> {
    pub fn add<J, RhsGrad, OutGrad>(
        &self,
        rhs: &Tensor<F, J, RhsGrad>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: RequiresGradTrait<F>,
        F: Copy + Add<Output = F> + Zero + Clone,
        for<'a> ArrayRef<'a, F, T>: OpsAdd<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsAdd<F> + OpsBroadcast<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> =
            storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, F, J> =
            storage.get_as_array_ref(rhs.array_idx, ContiguousType::Arr)?;

        let array = OpsAdd::add(&lhs_array, &rhs_array)?;
        let shape = array.shape.to_vec();

        let (lhs_broadcast, rhs_broadcast) = broadcast_detect(lhs_array.shape, rhs_array.shape);

        let array_idx = storage.push(ElementType::Arr(array))?;
        let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;

        let record_label = RecordLabel::Add(
            (self.array_idx, self.grad_idx, lhs_broadcast),
            (rhs.array_idx, rhs.grad_idx, rhs_broadcast),
            grad_idx,
        );
        self.record.borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        ))
    }
}

pub fn add_backward<F>(
    gradient_idx: Option<StorageType>,
    lhs_grad: Option<StorageType>,
    lhs_broadcast_dim: Option<&Vec<usize>>,
    rhs_grad: Option<StorageType>,
    rhs_broadcast_dim: Option<&Vec<usize>>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Clone + AddAssign + Copy + Zero + Sum<&'a F>,
{
    if let Some(gradient_idx) = gradient_idx {
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        };

        let gradient = storage
            .get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?
            .into_array();

        if let Some(lhs_grad) = lhs_grad {
            if !check_no_grad_or_time_not_match(lhs_grad, storage)? {
                let mut lhs_gradient: ArrayRefMut<'_, F, View> =
                    storage.get_as_array_ref_mut(lhs_grad, ContiguousType::Grad)?;
                match lhs_broadcast_dim {
                    Some(dim) => {
                        let gradient = OpsSum::sum_axis(&gradient, dim, true)?;
                        let to_shape = gradient.to_shape(lhs_gradient.shape)?;
                        lhs_gradient.add_assign(&to_shape)?
                    }
                    None => lhs_gradient.add_assign(&gradient)?,
                }
            };
        }

        if let Some(rhs_grad) = rhs_grad {
            if !check_no_grad_or_time_not_match(rhs_grad, storage)? {
                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_grad, ContiguousType::Grad)?;

                match rhs_broadcast_dim {
                    Some(dim) => {
                        let gradient = gradient.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(rhs_gradient.shape)?;
                        rhs_gradient.add_assign(&to_shape)?;
                    }
                    None => rhs_gradient.add_assign(&gradient)?,
                }
            };
        }
    }

    Ok(())
}
