use crate::prelude::*;
use num_traits::{Float, One};
use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    iter::Sum,
    marker::PhantomData,
    ops::AddAssign,
    rc::Rc,
};

pub struct Tensor<F, T, G> {
    pub(crate) record: Rc<RefCell<Vec<RecordLabel<F>>>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) array_idx: StorageType,
    pub(crate) grad_idx: Option<StorageType>,
    pub(crate) shape: Vec<usize>,
    pub(crate) _array_type: PhantomData<(T, G)>,
}

impl<F, T, G> Tensor<F, T, G> {
    pub fn array_to_string(&self) -> Result<String, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Debug + Copy,
    {
        let string = format!(
            "{}",
            self.storage
                .borrow()
                .get_as_array_ref::<T>(self.array_idx, ContiguousType::Arr)?
        );

        Ok(string)
    }

    pub fn grad_to_string(&self) -> Result<String, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Debug + Copy,
    {
        let string = format!(
            "{}",
            self.storage.borrow().get_as_array_ref::<T>(
                self.grad_idx.ok_or(PzeudoErr::ReqGradErr(format!(
                    "Tensor::grad_to_string. Tensor with NoGrad status"
                )))?,
                ContiguousType::Grad
            )?
        );

        Ok(string)
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn backward(&self) -> Result<(), PzeudoErr>
    where
        F: Display,
        ArrayStorage<F>: StorageF32F64,
        for<'a> F: Clone + One + AddAssign + Float + Sum<&'a F>,
        for<'a> ArrayRefMut<'a, F, T>: ArrayTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();
        if let Some(grad_idx) = self.grad_idx {
            let mut grad = storage.get_as_array_ref_mut::<T>(grad_idx, ContiguousType::Grad)?;
            let len = grad.shape.iter().product::<usize>();
            for i in 0..len {
                *grad.mut_linear_index(i)? += F::one();
            }
        }

        let mut record = self.record.borrow_mut();

        for record in record.iter().rev() {
            record.backward(&mut storage)?;
        }

        record.clear();

        Ok(())
    }
}
