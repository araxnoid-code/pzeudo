use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::prelude::*;

pub struct TensorView<F> {
    pub(crate) array_idx: usize,
    pub(crate) grad_idx: Option<usize>,
    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
}
