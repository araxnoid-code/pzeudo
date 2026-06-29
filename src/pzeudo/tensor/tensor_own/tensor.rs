use std::{
    cell::RefCell,
    fmt::Display,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use ndarray::{ArrayD, ArrayViewD};

use crate::{Backward, BackwardLabel, TensorTrait};

pub struct Tensor<'a> {
    pub(crate) array: ArrayD<f32>,
    pub(crate) gradient: Option<Rc<RefCell<ArrayD<f32>>>>,
    pub(crate) backward_label: Option<Arc<BackwardLabel<'a>>>,
    pub(crate) label_ops: AtomicBool,
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
            label_ops: AtomicBool::new(false),
        }
    }

    pub fn from_array(array: ArrayD<f32>) -> Tensor<'a> {
        Self {
            gradient: Some(Rc::new(RefCell::new(ArrayD::<f32>::zeros(array.shape())))),
            array,
            backward_label: None,
            label_ops: AtomicBool::new(false),
        }
    }
}

impl<'a> TensorTrait<'a> for Tensor<'a> {
    fn get_label_ops(&self) -> bool {
        self.label_ops.load(Ordering::Relaxed)
    }

    fn set_label_ops(&self, label: bool) {
        self.label_ops.store(label, Ordering::Relaxed);
    }

    fn get_array_view(&'a self) -> ArrayViewD<'a, f32> {
        self.array.view()
    }

    fn get_share_gradient(&self) -> Option<Rc<RefCell<ArrayD<f32>>>> {
        self.gradient.clone()
    }

    fn get_share_backward_label(&'a self) -> Option<Arc<BackwardLabel<'a>>> {
        self.backward_label.clone()
    }

    fn set_gradient_ones(&self) -> Result<(), &str> {
        if let Some(grad) = &self.gradient {
            *grad.borrow_mut() = ArrayD::<f32>::ones(self.array.shape());
            return Ok(());
        } else {
            return Err("Gradient Disable");
        }
    }
}

impl<'a> Display for Tensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{}", self.array))
    }
}

impl<'a> Backward<'a> for Tensor<'a> {}
