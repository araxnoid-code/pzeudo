use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, TensorTrait};

pub struct Tensor<'a> {
    array: ArrayD<f32>,
    gradient: Option<Rc<RefCell<ArrayD<f32>>>>,
    backward_label: Option<BackwardLabel<'a>>,
}

impl<'a> Tensor<'a> {
    pub fn new(
        array: ArrayD<f32>,
        gradient: Option<ArrayD<f32>>,
        backward_label: Option<BackwardLabel<'a>>,
    ) -> Tensor<'a> {
        Self {
            array,
            gradient: gradient.map(|grad| Rc::new(RefCell::new(grad))),
            backward_label,
        }
    }
}

impl<'a> TensorTrait<'a> for Tensor<'a> {
    fn array_view(&'a self) -> ArrayViewD<'a, f32> {
        self.array.view()
    }

    fn share_gradient(&self) -> Option<Rc<RefCell<ArrayD<f32>>>> {
        self.gradient.clone()
    }
}
