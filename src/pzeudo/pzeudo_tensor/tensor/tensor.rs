use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::prelude::*;

pub struct Tensor<F, T> {
    pub(crate) array_idx: usize,
    pub(crate) array_metadata: TensorMetadata,

    pub(crate) grad_idx: Option<usize>,
    pub(crate) grad_metadata: Option<TensorMetadata>,

    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,

    pub(crate) _tensor_type: PhantomData<T>,
}
