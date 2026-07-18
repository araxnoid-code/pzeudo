use std::{cell::RefCell, rc::Rc};

use num_traits::Zero;

use crate::prelude::*;

impl<F> Tensor<F> {
    pub fn new(
        array: Array<F>,
        gradient: Option<Array<F>>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel>>>,
    ) -> Result<Tensor<F>, PzeudoErr> {
        let mut storage_mut = storage.borrow_mut();

        let array_idx = storage_mut.push(array)?;
        let grad_idx =
            gradient.map_or(Ok(None), |grad| storage_mut.push(grad).map(|idx| Some(idx)));

        drop(storage_mut);

        let tensor: Tensor<F> = Self {
            array_idx,
            grad_idx: grad_idx?,
            storage,
            record,
        };

        Ok(tensor)
    }

    pub fn from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel>>>,
    ) -> Result<Self, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = storage.borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let gradient: Array<F> = Array::zeros(shape);
        let array_idx = borrow_storage.push(array)?;

        let grad_idx = Some(borrow_storage.push(gradient)?);
        drop(borrow_storage);

        let tensor = Tensor {
            array_idx,
            grad_idx,
            storage,
            record,
        };

        Ok(tensor)
    }

    pub fn view(&self) -> Result<TensorView<F>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut storage = self.storage.borrow_mut();
        let array = storage.get_element(self.array_idx)?;
        let tensor_metadata = TensorMetadata {
            offset: array.offset,
            shape: array.shape.clone(),
            stride: array.stride.clone(),
        };

        let grad = Array::<F>::zeros(&array.shape);
        let grad_idx = Some(storage.push(grad)?);

        drop(storage);

        let view = TensorView {
            array_idx: self.array_idx,
            grad_idx,
            metadata: tensor_metadata,
            record: self.record.clone(),
            storage: self.get_storage().clone(),
        };

        Ok(view)
    }
}
