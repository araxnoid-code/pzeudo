use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayViewD};

use crate::BackwardLabel;

pub trait TensorTrait<'a> {
    fn set_label_ops(&self, label: bool);
    fn get_label_ops(&self) -> bool;

    fn get_array_view(&'a self) -> ArrayViewD<'a, f32>;
    fn get_share_gradient(&self) -> Option<Rc<RefCell<ArrayD<f32>>>>;
    fn get_share_backward_label(&'a self) -> Option<Arc<BackwardLabel<'a>>>;
    fn set_gradient_ones(&self) -> Result<(), &str>;
}
