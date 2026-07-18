use std::{cell::RefCell, rc::Rc};

use num_traits::Zero;

use crate::prelude::*;

impl<F> Tensor<F, Contiguous> {
    pub fn new(
        array: Array<F>,
        gradient: Option<Array<F>>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr> {
        let mut storage_mut = storage.borrow_mut();

        let array_idx = storage_mut.push(array.data)?;
        let arr_metadata = TensorMetadata {
            offset: array.offset,
            shape: array.shape,
            stride: array.stride,
        };

        let (grad_idx, grad_metadata) = gradient.map_or((Ok(None), None), |grad| {
            let arr_metadata = TensorMetadata {
                offset: grad.offset,
                shape: grad.shape,
                stride: grad.stride,
            };

            (
                storage_mut.push(grad.data).map(|idx| Some(idx)),
                Some(arr_metadata),
            )
        });

        drop(storage_mut);

        let tensor: Tensor<F, Contiguous> = Self {
            array_idx,
            array_metadata: arr_metadata,
            grad_idx: grad_idx?,
            grad_metadata,
            storage,
            record,
            _tensor_type: Default::default(),
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

        let arr_metadata = TensorMetadata {
            offset: array.offset,
            shape: array.shape,
            stride: array.stride,
        };
        let array_idx = borrow_storage.push(array.data)?;

        let grad_metadata = TensorMetadata {
            offset: gradient.offset,
            shape: gradient.shape,
            stride: gradient.stride,
        };
        let grad_idx = Some(borrow_storage.push(gradient.data)?);
        drop(borrow_storage);

        let tensor = Tensor {
            array_idx,
            array_metadata: arr_metadata,
            grad_idx,
            grad_metadata: Some(grad_metadata),
            storage,
            record,
            _tensor_type: Default::default(),
        };

        Ok(tensor)
    }
}
