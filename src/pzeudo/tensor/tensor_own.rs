use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, Neg},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use ndarray::{ArrayD, ArrayViewD};
use num_traits::{Float, One, Zero};

use crate::{Backward, BackwardLabel, PzeudoOpsAdd, TensorTrait};

pub struct Tensor<'backward_label, F> {
    pub(crate) array: ArrayD<F>,
    pub(crate) gradient: Option<Rc<RefCell<ArrayD<F>>>>,
    pub(crate) backward_label: Option<Arc<BackwardLabel<'backward_label, F>>>,
    pub(crate) label_ops: AtomicBool,
}

impl<'bacward_label, F> Tensor<'bacward_label, F>
where
    F: Clone + Zero + One + Float + Add<Output = F> + Copy,
{
    pub fn new(
        array: ArrayD<F>,
        gradient: Option<ArrayD<F>>,
        backward_label: Option<BackwardLabel<'bacward_label, F>>,
    ) -> Tensor<'bacward_label, F> {
        Self {
            array,
            gradient: gradient.map(|grad| Rc::new(RefCell::new(grad))),
            backward_label: backward_label.map(|label| Arc::new(label)),
            label_ops: AtomicBool::new(false),
        }
    }

    pub fn from_array(array: ArrayD<F>) -> Tensor<'bacward_label, F> {
        Self {
            gradient: Some(Rc::new(RefCell::new(ArrayD::<F>::zeros(array.shape())))),
            array,
            backward_label: None,
            label_ops: AtomicBool::new(false),
        }
    }
}

impl<'backward_label, F> TensorTrait<'backward_label, F> for Tensor<'backward_label, F>
where
    F: Clone + One + Zero,
{
    fn get_label_ops(&self) -> bool {
        self.label_ops.load(Ordering::Relaxed)
    }

    fn set_label_ops(&self, label: bool) {
        self.label_ops.store(label, Ordering::Relaxed);
    }

    fn get_array_view(&self) -> ArrayViewD<'_, F> {
        self.array.view()
    }

    fn get_share_gradient(&self) -> Option<Rc<RefCell<ArrayD<F>>>> {
        self.gradient.clone()
    }

    fn get_share_backward_label(&self) -> Option<Arc<BackwardLabel<'backward_label, F>>> {
        self.backward_label.clone()
    }

    fn set_gradient_ones(&self) -> Result<(), &str> {
        if let Some(grad) = &self.gradient {
            *grad.borrow_mut() = ArrayD::<F>::ones(self.array.shape());
            return Ok(());
        } else {
            return Err("Gradient Disable");
        }
    }
}

impl<'bacward_label, F> PzeudoOpsAdd<'bacward_label, F> for Tensor<'bacward_label, F> where
    F: Add<Output = F> + Copy + Zero + Float
{
}

// impl<'bacward_label, F> PzeudoOpsSub<'bacward_label> for Tensor<'bacward_label, F> {}
// impl<'bacward_label, F> PzeudoOpsMul<'bacward_label> for Tensor<'bacward_label, F> {}
// impl<'bacward_label, F> PzeudoOpsDiv<'bacward_label> for Tensor<'bacward_label, F> {}

impl<'bacward_label, F> Backward<'bacward_label, F> for Tensor<'bacward_label, F> where
    F: AddAssign<F>
        + Clone
        + Copy
        + Zero
        + Neg<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + One
{
}

impl<'bacward_label, F> Display for Tensor<'bacward_label, F>
where
    F: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{}", self.array))
    }
}
