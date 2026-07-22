use std::{cell::RefCell, marker::PhantomData, rc::Rc};

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

        let array_idx = storage_mut.push(ElementType::Contiguous(array, ContiguousType::Arr))?;
        let grad_idx = gradient.map_or(Ok(None), |grad| {
            storage_mut
                .push(ElementType::Contiguous(grad, ContiguousType::Grad))
                .map(|idx| Some(idx))
        });

        drop(storage_mut);

        let tensor = Self {
            array_idx,
            grad_idx: grad_idx?,
            storage,
            record,
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = storage.borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let gradient: Array<F> = Array::zeros(shape);
        let array_idx = borrow_storage.push(ElementType::Contiguous(array, ContiguousType::Arr))?;

        let grad_idx =
            Some(borrow_storage.push(ElementType::Contiguous(gradient, ContiguousType::Grad))?);
        drop(borrow_storage);

        let tensor = Tensor {
            array_idx,
            grad_idx,
            storage,
            record,
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn permanent_from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = storage.borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let gradient: Array<F> = Array::zeros(shape);
        let update_able_idx = borrow_storage.push_permanent_tensor(array, gradient)?;

        drop(borrow_storage);

        let tensor = Tensor {
            array_idx: update_able_idx,
            grad_idx: Some(update_able_idx),
            storage,
            record,
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn from_array(
        array: Array<F>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
        record: Rc<RefCell<Vec<RecordLabel>>>,
    ) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut storage_borrow_mut = storage.borrow_mut();

        let gradient = Array::<F>::zeros(&array.shape);
        let array_idx =
            storage_borrow_mut.push(ElementType::Contiguous(array, ContiguousType::Arr))?;
        let grad_idx =
            Some(storage_borrow_mut.push(ElementType::Contiguous(gradient, ContiguousType::Grad))?);
        drop(storage_borrow_mut);

        Ok(Tensor {
            array_idx,
            grad_idx,
            record: record,
            storage: storage,
            _array_type: PhantomData::default(),
        })
    }

    pub fn view(&self) -> Result<Tensor<F, View>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut storage = self.storage.borrow_mut();
        let array: ArrayRef<'_, F, Contiguous> =
            storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;

        let tensor_metadata = TensorMetadata {
            offset: array.offset,
            shape: array.shape.to_vec(),
            stride: array.stride.to_vec(),
        };

        let grad = Array::<F>::zeros(&array.shape);
        let grad_idx = Some(storage.push(ElementType::Contiguous(grad, ContiguousType::Grad))?);

        let view_idx = storage.push(ElementType::View(self.array_idx, tensor_metadata))?;

        drop(storage);
        let view = Tensor {
            array_idx: view_idx,
            grad_idx,
            record: self.record.clone(),
            storage: self.storage.clone(),
            _array_type: Default::default(),
        };

        Ok(view)
    }
}
