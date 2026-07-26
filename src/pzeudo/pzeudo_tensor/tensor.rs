use crate::prelude::*;
use num_traits::{Float, One};
use std::{cell::RefCell, iter::Sum, marker::PhantomData, ops::AddAssign, rc::Rc};

pub struct Tensor<F, T> {
    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>, // 8
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>, // 8
    pub(crate) array_idx: StorageType,                // 16
    pub(crate) grad_idx: Option<StorageType>,         // 16
    pub(crate) _array_type: PhantomData<T>,           // 0
}

impl<F, T> Tensor<F, T> {
    pub fn backward(&self) -> Result<(), PzeudoErr>
    where
        ArrayStorage<F>: StorageF32F64,
        for<'a> F: Clone + One + AddAssign + Float + Sum<&'a F>,
        for<'a> ArrayRef<'a, F, Contiguous>: OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, View>: OpsBroadcast<F>,
    {
        let mut storage = self.storage.borrow_mut();
        if let Some(grad_idx) = self.grad_idx {
            let grad = storage.get_grad_element_mut(grad_idx)?;
            grad.to_ones();
        }

        let mut record = self.record.borrow_mut();

        for record in record.iter().rev() {
            record.backward(&mut storage)?;
        }

        record.clear();

        Ok(())
    }
}
