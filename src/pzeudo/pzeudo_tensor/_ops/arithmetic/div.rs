use std::ops::{AddAssign, Div, Neg};

use num_traits::{Float, One, Zero, one};

use crate::prelude::*;

pub trait TensorDivOps<F, T>: TensorTrait<F, T> {
    fn div<Rhs, J>(&self, rhs: &Rhs) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Copy + Div<Output = F> + Zero + Clone,
        Rhs: TensorTrait<F, J>,
        for<'a> ArrayRef<'a, F, T>: OpsDiv<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsDiv<F> + OpsBroadcast<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> = storage.get_as_array_ref(self.get_array_idx())?;
        let rhs_array: ArrayRef<'_, F, J> = storage.get_as_array_ref(rhs.get_array_idx())?;

        let array = OpsDiv::div(&lhs_array, &rhs_array)?;
        let (lhs_broadcast, rhs_broadcast) = broadcast_detech(lhs_array.shape, rhs_array.shape);

        let grad = Array::<F>::zeros(&array.shape);
        let array_idx = storage.push(ElementType::Contiguous(array))?;
        let grad_idx = Some(storage.push(ElementType::Contiguous(grad))?);

        let record_label = RecordLabel::Div(
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

impl<F, T> TensorDivOps<F, T> for Tensor<F, T> {}

pub fn div_backward<F>(
    gradient_idx: Option<usize>,
    lhs: usize,
    rhs: usize,
    lhs_grad: Option<usize>,
    rhs_grad: Option<usize>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Clone + AddAssign + Copy + Div<Output = F> + One + Neg<Output = F> + Float,
    for<'a> ArrayRef<'a, F, Contiguous>: OpsBroadcast<F>,
{
    // f(lhs, rhs) = lhs / rhs
    if let Some(gradient_idx) = gradient_idx {
        let gradien = storage.get_as_array_ref::<Contiguous>(gradient_idx)?;

        if let Some(lhs_grad) = lhs_grad {
            // df(lhs, rhs)/dlhs = 1/rhs * gradient
            let rhs_value: ArrayRef<'_, F, View> = storage.get_as_array_ref(rhs)?;
            let grad = rhs_value.scalar_div(one())?.mul(&gradien)?;

            let mut lhs_grad = storage.get_as_array_ref_mut(lhs_grad)?;
            lhs_grad.add_assign(&grad)?;
        }

        let gradien = storage.get_as_array_ref::<Contiguous>(gradient_idx)?;
        if let Some(rhs_grad) = rhs_grad {
            // df(lhs, rhs)/drhs = -lhs/rhs^2 * gradient
            let rhs_value: ArrayRef<'_, F, View> = storage.get_as_array_ref(rhs)?;
            let lhs_value: ArrayRef<'_, F, View> = storage.get_as_array_ref(lhs)?;
            let grad = (lhs_value.neg()? / rhs_value.powi(2)?).mul(&gradien)?;

            let mut lhs_gradient = storage.get_as_array_ref_mut(rhs_grad)?;
            lhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}
