use std::ops::{AddAssign, Neg, Sub};

use num_traits::Zero;

use crate::prelude::*;

pub trait TensorSubOps<F, T>: TensorTrait<F, T> {
    fn sub<Rhs, J>(&self, rhs: &Rhs) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Copy + Sub<Output = F> + Zero + Clone,
        Rhs: TensorTrait<F, J>,
        for<'a> ArrayRef<'a, F, T>: OpsSub<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsSub<F> + OpsBroadcast<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> = storage.get_as_array_ref(self.get_array_idx())?;
        let rhs_array: ArrayRef<'_, F, J> = storage.get_as_array_ref(rhs.get_array_idx())?;

        let array = OpsSub::sub(&lhs_array, &rhs_array)?;
        let (lhs_broadcast, rhs_broadcast) = broadcast_detech(lhs_array.shape, rhs_array.shape);

        let grad = Array::<F>::zeros(&array.shape);
        let array_idx = storage.push(ElementType::Contiguous(array))?;
        let grad_idx = Some(storage.push(ElementType::Contiguous(grad))?);

        let record_label = RecordLabel::Sub(
            (self.get_array_idx(), self.get_grad_idx(), lhs_broadcast),
            (rhs.get_array_idx(), rhs.get_grad_idx(), rhs_broadcast),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        let tensor = Tensor {
            array_idx,
            grad_idx,
            record: self.get_record().clone(),
            storage: self.get_storage().clone(),
            _array_type: Default::default(),
        };

        Ok(tensor)
    }
}

impl<F, T> TensorSubOps<F, T> for Tensor<F, T> {}

pub fn sub_backward<F>(
    gradient_idx: Option<usize>,
    lhs_grad: Option<usize>,
    rhs_grad: Option<usize>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Clone + AddAssign + Copy + Neg<Output = F>,
{
    // f(lhs, rhs) = lhs - rhs
    if let Some(gradient_idx) = gradient_idx {
        let gradien = storage
            .get_as_array_ref::<Contiguous>(gradient_idx)?
            .into_array();

        if let Some(lhs_grad) = lhs_grad {
            // df(lhs, rhs)/dlhs = 1 * gradient
            let mut lhs_grad = storage.get_as_array_ref_mut(lhs_grad)?;
            lhs_grad.add_assign(&gradien)?;
        }

        // let gradien = storage.get_as_array_ref::<Contiguous>(gradient_idx)?;
        if let Some(rhs_grad) = rhs_grad {
            // df(lhs, rhs)/drhs = -1 * gradient
            let grad = gradien.neg()?;

            let mut lhs_gradient = storage.get_as_array_ref_mut(rhs_grad)?;
            lhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}
