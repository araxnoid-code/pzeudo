use crate::prelude::*;
use num_traits::Zero;
use std::ops::{Add, AddAssign};

pub trait TensorAddOps<F, T>: TensorTrait<F, T> {
    fn add<Rhs, J>(&self, rhs: &Rhs) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Copy + Add<Output = F> + Zero + Clone,
        Rhs: TensorTrait<F, J>,
        for<'a> ArrayRef<'a, F, T>: OpsAdd<F>,
        for<'a> ArrayRef<'a, F, J>: OpsAdd<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> = storage.get_as_array_ref(self.get_array_idx())?;
        let rhs_array: ArrayRef<'_, F, J> = storage.get_as_array_ref(rhs.get_array_idx())?;

        let array = OpsAdd::add(&lhs_array, &rhs_array)?;
        let grad = Array::<F>::zeros(&array.shape);

        let array_idx = storage.push(ElementType::Contiguous(array))?;
        let grad_idx = Some(storage.push(ElementType::Contiguous(grad))?);

        let record_label = RecordLabel::Add(
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

impl<F, T> TensorAddOps<F, T> for Tensor<F, T> {}

fn add_backward<F>(
    gradient_idx: Option<usize>,
    lhs_grad: Option<usize>,
    rhs_grad: Option<usize>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Clone + AddAssign + Copy,
{
    if let Some(gradient_idx) = gradient_idx {
        let gradien: Array<F> = storage
            .get_as_array_ref::<Contiguous>(gradient_idx)?
            .into_array();

        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_gradient = storage.get_as_array_ref_mut(lhs_grad)?;
            lhs_gradient.add_assign(&gradien)?;
        }

        if let Some(rhs_grad) = rhs_grad {
            let mut lhs_gradient = storage.get_as_array_ref_mut(rhs_grad)?;
            lhs_gradient.add_assign(&gradien)?;
        }
    }
    Ok(())
}
