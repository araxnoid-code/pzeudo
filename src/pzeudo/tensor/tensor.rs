use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::{BackwardLabel, GradientStorage};

pub trait TensorTrait<'backward_label, F, Grad, GradStorage: GradientStorage<Grad>> {
    fn set_gradient_ones(&self) -> Result<(), &str>;

    fn set_label_ops(&self, label: bool);
    fn get_label_ops(&self) -> bool;

    fn get_array_view(&self) -> ArrayViewD<'_, F>;
    fn get_share_gradient(&self) -> Option<usize>;
    fn get_share_backward_label(&self) -> Option<&BackwardLabel<'backward_label, F>>;

    fn get_storage(&self) -> &Rc<RefCell<GradStorage>>;
}
