use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayViewD};

use crate::BackwardLabel;

pub trait TensorTrait<'backward_label, F> {
    fn set_gradient_ones(&self) -> Result<(), &str>;

    fn set_label_ops(&self, label: bool);
    fn get_label_ops(&self) -> bool;

    fn get_array_view(&self) -> ArrayViewD<'_, F>;
    fn get_share_gradient(&self) -> Option<Rc<RefCell<ArrayD<F>>>>;
    fn get_share_backward_label(&self) -> Option<Arc<BackwardLabel<'backward_label, F>>>;
}
