use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use num_traits::{One, Zero};

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub(crate) fn _new(
        array_idx: StorageType,
        grad_idx: Option<StorageType>,
        shape: Vec<usize>,
        record_status: Option<RecordStatus>,
        record: Rc<RefCell<Record<F>>>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
    ) -> Tensor<F, T, G> {
        Self {
            array_idx,
            shape,
            grad_idx,
            record_status,
            record,
            storage,
            _array_type: Default::default(),
        }
    }

    pub fn new(
        array_idx: StorageType,
        grad_idx: Option<StorageType>,
        shape: Vec<usize>,
        module: &ModuleBuilder<F>,
    ) -> Tensor<F, T, G> {
        Self {
            array_idx,
            shape,
            grad_idx,
            record_status: None,
            record: module.record.clone(),
            storage: module.storage.clone(),
            _array_type: Default::default(),
        }
    }
}

impl<F, G> Tensor<F, Contiguous, G>
where
    G: ReqGradTrait<F>,
{
    pub fn ones(
        shape: &[usize],
        module: &ModuleBuilder<F>,
        requires_grad: G,
    ) -> Result<Tensor<F, Contiguous, G>, PzeudoErr>
    where
        F: Clone + One,
    {
        let mut borrow_storage = module.get_storage().borrow_mut();
        let array: Array<F> = Array::ones(shape);

        let array_idx = borrow_storage.push(ElementType::Arr(array))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(shape, &mut borrow_storage)?;

        let tensor = Tensor {
            array_idx,
            grad_idx,
            storage: module.storage.clone(),
            record: module.record.clone(),
            record_status: None,
            shape: shape.to_vec(),
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        module: &ModuleBuilder<F>,
        requires_grad: G,
    ) -> Result<Tensor<F, Contiguous, G>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = module.get_storage().borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let array_idx = borrow_storage.push(ElementType::Arr(array))?;

        let grad_idx = requires_grad.into_zeros_grad_storage(shape, &mut borrow_storage)?;
        drop(borrow_storage);

        let tensor = Tensor {
            array_idx,
            grad_idx,
            storage: module.storage.clone(),
            record: module.record.clone(),
            record_status: None,
            shape: shape.to_vec(),
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn from_array(
        array: Array<F>,
        module: &ModuleBuilder<F>,
        requires_grad: G,
    ) -> Result<Tensor<F, Contiguous, G>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let shape = array.shape.to_vec();
        let mut storage_borrow_mut = module.get_storage().borrow_mut();

        let grad_idx =
            requires_grad.into_zeros_grad_storage(&array.shape, &mut storage_borrow_mut)?;
        let array_idx = storage_borrow_mut.push(ElementType::Arr(array))?;
        drop(storage_borrow_mut);

        Ok(Tensor {
            array_idx,
            grad_idx,
            record: module.get_record().clone(),
            storage: module.get_storage().clone(),
            record_status: None,
            shape,
            _array_type: PhantomData::default(),
        })
    }
}

#[allow(unused)]
impl<F, ReqGrad> Tensor<F, Contiguous, ReqGrad>
where
    ReqGrad: ReqGradTrait<F>,
{
    pub fn param_ones(
        shape: &[usize],
        module: &ModuleBuilder<F>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: Clone + One,
    {
        let mut borrow_storage = module.get_storage().borrow_mut();

        let array: Array<F> = Array::ones(shape);
        let gradient = ReqGrad::zeros_grad(shape);
        let params_idx = borrow_storage.push_param_tensor(array, gradient);

        let tensor = Tensor {
            array_idx: params_idx,
            grad_idx: Some(params_idx),
            storage: module.get_storage().clone(),
            record: module.get_record().clone(),
            shape: shape.to_vec(),
            record_status: None,
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn param_zeros(
        shape: &[usize],
        module: &ModuleBuilder<F>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = module.get_storage().borrow_mut();

        let array: Array<F> = Array::zeros(shape);
        let gradient = ReqGrad::zeros_grad(shape);
        let params_idx = borrow_storage.push_param_tensor(array, gradient);

        let tensor = Tensor {
            array_idx: params_idx,
            grad_idx: Some(params_idx),
            storage: module.get_storage().clone(),
            record: module.get_record().clone(),
            shape: shape.to_vec(),
            record_status: None,
            _array_type: Default::default(),
        };

        Ok(tensor)
    }

    pub fn param_from_vector_with_shape(
        vec: &[F],
        shape: &[usize],
        module: &ModuleBuilder<F>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut borrow_storage = module.get_storage().borrow_mut();

        let array = Array::from_vector_with_shape(vec, shape)?;
        let gradient = ReqGrad::zeros_grad(&shape);
        let update_able_idx = borrow_storage.push_param_tensor(array, gradient);

        drop(borrow_storage);

        let tensor = Tensor {
            array_idx: update_able_idx,
            grad_idx: Some(update_able_idx),
            storage: module.get_storage().clone(),
            record: module.get_record().clone(),
            shape: shape.to_vec(),
            record_status: None,
            _array_type: Default::default(),
        };

        Ok(tensor)
    }
}
