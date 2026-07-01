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

use crate::{
    Backward, BackwardLabel, PzeudoOpsAdd, PzeudoOpsDiv, PzeudoOpsMul, PzeudoOpsSub, TensorTrait,
};

pub struct TensorF32<'backward_label> {
    pub(crate) array: ArrayD<f32>,
    pub(crate) gradient: Option<Rc<RefCell<ArrayD<f32>>>>,
    pub(crate) backward_label: Option<Arc<BackwardLabel<'backward_label>>>,
    pub(crate) label_ops: AtomicBool,
}

impl<'bacward_label> TensorF32<'bacward_label> {
    pub fn new(
        array: ArrayD<f32>,
        gradient: Option<ArrayD<f32>>,
        backward_label: Option<BackwardLabel<'bacward_label>>,
    ) -> TensorF32<'bacward_label> {
        Self {
            array,
            gradient: gradient.map(|grad| Rc::new(RefCell::new(grad))),
            backward_label: backward_label.map(|label| Arc::new(label)),
            label_ops: AtomicBool::new(false),
        }
    }

    pub fn from_array(array: ArrayD<f32>) -> TensorF32<'bacward_label> {
        Self {
            gradient: Some(Rc::new(RefCell::new(ArrayD::<f32>::zeros(array.shape())))),
            array,
            backward_label: None,
            label_ops: AtomicBool::new(false),
        }
    }
}

impl<'backward_label> TensorTrait<'backward_label, f32> for TensorF32<'backward_label> {
    fn get_label_ops(&self) -> bool {
        self.label_ops.load(Ordering::Relaxed)
    }

    fn set_label_ops(&self, label: bool) {
        self.label_ops.store(label, Ordering::Relaxed);
    }

    fn get_array_view(&self) -> ArrayViewD<'_, f32> {
        self.array.view()
    }

    fn get_share_gradient(&self) -> Option<Rc<RefCell<ArrayD<f32>>>> {
        self.gradient.clone()
    }

    fn get_share_backward_label(&self) -> Option<Arc<BackwardLabel<'backward_label>>> {
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

impl<'bacward_label> PzeudoOpsAdd<'bacward_label> for TensorF32<'bacward_label> {}
impl<'bacward_label> PzeudoOpsSub<'bacward_label> for TensorF32<'bacward_label> {}
impl<'bacward_label> PzeudoOpsMul<'bacward_label> for TensorF32<'bacward_label> {}
impl<'bacward_label> PzeudoOpsDiv<'bacward_label> for TensorF32<'bacward_label> {}

impl<'bacward_label> Backward<'bacward_label, f32> for TensorF32<'bacward_label> {}

impl<'bacward_label> Display for TensorF32<'bacward_label> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{}", self.array))
    }
}
