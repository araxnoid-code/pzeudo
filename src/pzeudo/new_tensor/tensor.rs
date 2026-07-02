use std::{cell::RefCell, rc::Rc};

use crate::TensorAble;

pub struct Tensor<T: TensorAble> {
    pub array: T,
    pub gradient: Option<Rc<RefCell<T>>>,
}
