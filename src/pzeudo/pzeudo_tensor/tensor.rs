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
    pub(crate) record: Rc<RefCell<Record<F>>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) array_idx: StorageType,
    pub(crate) grad_idx: Option<StorageType>,
    pub(crate) shape: Vec<usize>,
    pub(crate) record_status: Option<RecordStatus>,
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
                *grad.linear_index_mut(i)? += F::one();
            }
            storage.set_grad_update(grad_idx, true)?;
        }

        let mut record = self.record.borrow_mut();

        for (record, skip) in record.record.iter().zip(record.skip.iter()).rev() {
            if *skip {
                continue;
            }
            record.backward(&mut storage)?;
        }

        record.clear();

        Ok(())
    }

    pub fn value_vec_eq(&self, vector: &[F]) -> Result<(), PzeudoErr>
    where
        F: Copy + Debug + PartialEq,
    {
        let len = self.shape.iter().product::<usize>();
        if len != vector.len() {
            return Err(PzeudoErr::TensorErr(format!(
                "Tensor::value_vec_eq. The length of vector {} is not equal to the length of tensor {}.",
                vector.len(),
                len
            )));
        }

        let storage = self.storage.borrow();
        let array = storage
            .get_as_array_ref::<View>(self.array_idx, ContiguousType::Arr)?
            .into_array()?;

        if array.data != vector {
            return Err(PzeudoErr::TensorErr(format!(
                "Tensor::value_vec_eq. The vector array and the input vector are not the same."
            )));
        }

        Ok(())
    }

    pub fn grad_vec_eq(&self, vector: &[F]) -> Result<(), PzeudoErr>
    where
        F: Copy + Debug + PartialEq,
    {
        let len = self.shape.iter().product::<usize>();
        if len != vector.len() {
            return Err(PzeudoErr::TensorErr(format!(
                "Tensor::grad_vec_eq. The length of vector {} is not equal to the length of tensor {}.",
                vector.len(),
                len
            )));
        }

        let storage = self.storage.borrow();
        let array = storage
            .get_as_array_ref::<View>(
                self.grad_idx.ok_or(PzeudoErr::TensorErr(format!(
                    "Tensor::grad_vec_eq. Tensor NoGrad."
                )))?,
                ContiguousType::Grad,
            )?
            .into_array()?;

        if array.data != vector {
            return Err(PzeudoErr::TensorErr(format!(
                "Tensor::grad_vec_eq. The vector array and the input vector are not the same."
            )));
        }
        Ok(())
    }
}
