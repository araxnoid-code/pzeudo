use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, TensorTrait};

pub struct Tensor<'a> {
    array: ArrayD<f32>,
    gradient: Option<Rc<RefCell<ArrayD<f32>>>>,
    backward_label: Option<Arc<BackwardLabel<'a>>>,
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
            backward_label: backward_label.map(|label| Arc::new(label)),
        }
    }
}

impl<'a> TensorTrait<'a> for Tensor<'a> {
    fn get_array_view(&'a self) -> ArrayViewD<'a, f32> {
        self.array.view()
    }

    fn get_share_gradient(&self) -> Option<Rc<RefCell<ArrayD<f32>>>> {
        self.gradient.clone()
    }

    fn get_share_backward_label(&'a self) -> Option<Arc<BackwardLabel<'a>>> {
        self.backward_label.clone()
    }
}
