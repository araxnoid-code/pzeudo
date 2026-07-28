use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use num_traits::Zero;

use crate::prelude::*;

impl<F, T> Tensor<F, T> {
    pub fn new(
        array_idx: StorageType,
        grad_idx: Option<StorageType>,
        shape: Vec<usize>,
        record: Rc<RefCell<Vec<RecordLabel<F>>>>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
    ) -> Tensor<F, T> {
        Self {
            array_idx,
            grad_idx,
            record,
            shape,
            storage,
            _array_type: Default::default(),
        }
    }
}

impl<F> Tensor<F, Contiguous> {
    pub fn from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel<F>>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = storage.borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let gradient: Array<F> = Array::zeros(shape);
        let array_idx = borrow_storage.push(ElementType::Arr(array))?;

        let grad_idx = Some(borrow_storage.push(ElementType::Grad(gradient))?);
        drop(borrow_storage);

        let tensor = Tensor {
            array_idx,
            grad_idx,
            storage,
            record,
            shape: shape.to_vec(),
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn permanent_from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel<F>>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = storage.borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let gradient: Array<F> = Array::zeros(shape);
        let update_able_idx = borrow_storage.push_permanent_tensor(array, gradient);

        drop(borrow_storage);

        let tensor = Tensor {
            array_idx: update_able_idx,
            grad_idx: Some(update_able_idx),
            storage,
            record,
            shape: shape.to_vec(),
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn from_array(
        array: Array<F>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel<F>>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let shape = array.shape.to_vec();
        let mut storage_borrow_mut = storage.borrow_mut();

        let gradient = Array::<F>::zeros(&array.shape);
        let array_idx = storage_borrow_mut.push(ElementType::Arr(array))?;
        let grad_idx = Some(storage_borrow_mut.push(ElementType::Grad(gradient))?);
        drop(storage_borrow_mut);

        Ok(Tensor {
            array_idx,
            grad_idx,
            record: record,
            storage: storage,
            shape,
            _array_type: PhantomData::default(),
        })
    }
}
