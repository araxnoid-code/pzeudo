use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{Array, ArrayStorage, ArrayTrait};

pub struct Tensor<F> {
    pub(crate) array_idx: usize,
    pub(crate) grad_idx: Option<usize>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
}
