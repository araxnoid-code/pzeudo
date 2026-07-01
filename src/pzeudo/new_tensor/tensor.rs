use std::{cell::RefCell, rc::Rc};

use crate::TensorAble;

struct Tensor<T: TensorAble> {
    pub(crate) array: T,
    pub(crate) gradient: Option<Rc<RefCell<T>>>,
}
