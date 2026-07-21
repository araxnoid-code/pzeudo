use crate::prelude::*;
use num_traits::Zero;
use std::{
    iter::Sum,
    ops::{Add, AddAssign},
    println,
};

pub trait TensorAddOps<F, T>: TensorTrait<F, T> {
    fn add<Rhs, J>(&self, rhs: &Rhs) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Copy + Add<Output = F> + Zero + Clone,
        Rhs: TensorTrait<F, J>,
        for<'a> ArrayRef<'a, F, T>: OpsAdd<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsAdd<F> + OpsBroadcast<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> = storage.get_as_array_ref(self.get_array_idx())?;
        let rhs_array: ArrayRef<'_, F, J> = storage.get_as_array_ref(rhs.get_array_idx())?;

        let array = OpsAdd::add(&lhs_array, &rhs_array)?;
        let (lhs_broadcast, rhs_broadcast) = broadcast_detech(lhs_array.shape, rhs_array.shape);

        let grad = Array::<F>::zeros(&array.shape);
        let array_idx = storage.push(ElementType::Contiguous(array))?;
        let grad_idx = Some(storage.push(ElementType::Contiguous(grad))?);

        let record_label = RecordLabel::Add(
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

impl<F, T> TensorAddOps<F, T> for Tensor<F, T> {}

pub fn add_backward<F>(
    gradient_idx: Option<usize>,
    lhs_grad: Option<usize>,
    lhs_broadcast_dim: Option<&Vec<usize>>,
    rhs_grad: Option<usize>,
    rhs_broadcast_dim: Option<&Vec<usize>>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Clone + AddAssign + Copy + Zero + Sum<&'a F>,
{
    if let Some(gradient_idx) = gradient_idx {
        let gradient: Array<F> = storage
            .get_as_array_ref::<Contiguous>(gradient_idx)?
            .into_array();

        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_gradient = storage.get_as_array_ref_mut(lhs_grad)?;
            match lhs_broadcast_dim {
                Some(dim) => {
                    let gradient = OpsSum::sum_axis(&gradient, dim, true)?;
                    let to_shape = gradient.to_shape(lhs_gradient.shape)?;
                    lhs_gradient.add_assign(&to_shape)?
                }
                None => lhs_gradient.add_assign(&gradient)?,
            }
        }

        if let Some(rhs_grad) = rhs_grad {
            let mut rhs_gradient = storage.get_as_array_ref_mut(rhs_grad)?;
            match rhs_broadcast_dim {
                Some(dim) => {
                    let gradient = gradient.sum_axis(dim, true)?;
                    let to_shape = gradient.to_shape(rhs_gradient.shape)?;
                    rhs_gradient.add_assign(&to_shape)?
                }
                None => rhs_gradient.add_assign(&gradient)?,
            }
        }
    }
    Ok(())
}
