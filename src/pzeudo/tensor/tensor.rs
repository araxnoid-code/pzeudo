use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, ArrayViewD};

pub trait TensorTrait<'a> {
    fn array_view(&'a self) -> ArrayViewD<'a, f32>;
    fn share_gradient(&self) -> Option<Rc<RefCell<ArrayD<f32>>>>;
}
