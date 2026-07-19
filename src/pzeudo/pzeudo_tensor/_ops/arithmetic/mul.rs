use std::ops::{AddAssign, Mul};

use num_traits::Zero;

use crate::prelude::*;

pub trait TensorMulOps<F, T>: TensorTrait<F, T> {
    fn mul<Rhs, J>(&self, rhs: &Rhs) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Copy + Mul<Output = F> + Zero + Clone,
        Rhs: TensorTrait<F, J>,
        for<'a> ArrayRef<'a, F, T>: OpsMul<F>,
        for<'a> ArrayRef<'a, F, J>: OpsMul<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> = storage.get_as_array_ref(self.get_array_idx())?;
        let rhs_array: ArrayRef<'_, F, J> = storage.get_as_array_ref(rhs.get_array_idx())?;

        let array = OpsMul::mul(&lhs_array, &rhs_array)?;
        let grad = Array::<F>::zeros(&array.shape);

        let array_idx = storage.push(ElementType::Contiguous(array))?;
        let grad_idx = Some(storage.push(ElementType::Contiguous(grad))?);

        let record_label = RecordLabel::Mul(
            (self.get_array_idx(), self.get_grad_idx()),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
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

impl<F, T> TensorMulOps<F, T> for Tensor<F, T> {}

fn mul_backward<F, T>(
    gradient_idx: Option<usize>,
    lhs: usize,
    rhs: usize,
    lhs_grad: Option<usize>,
    rhs_grad: Option<usize>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Clone + AddAssign + Copy + Mul<Output = F>,
{
    // f(lhs, rhs) = lhs * rhs
    if let Some(gradient_idx) = gradient_idx {
        let gradien = storage.get_as_array_ref::<Contiguous>(gradient_idx)?;

        if let Some(lhs_grad) = lhs_grad {
            // df(lhs, rhs)/dlhs = rhs * gradient
            let rhs_value: ArrayRef<'_, F, View> = storage.get_as_array_ref(rhs)?;
            let grad = rhs_value.mul(&gradien)?;

            let mut lhs_grad = storage.get_as_array_ref_mut(lhs_grad)?;
            lhs_grad.add_assign(&grad)?;
        }

        let gradien = storage.get_as_array_ref::<Contiguous>(gradient_idx)?;
        if let Some(rhs_grad) = rhs_grad {
            // df(lhs, rhs)/drhs = lhs * gradient
            let lhs_value: ArrayRef<'_, F, View> = storage.get_as_array_ref(lhs)?;
            let grad = lhs_value.mul(&gradien)?;

            let mut lhs_gradient = storage.get_as_array_ref_mut(rhs_grad)?;
            lhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}
