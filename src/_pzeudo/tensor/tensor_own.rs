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

use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, linalg::Dot};
use num_traits::{Float, One, Zero};

use crate::{
    Backward, BackwardLabel, GradientStorage, PzeudoOpsAdd, PzeudoOpsDiv, PzeudoOpsMatmul2D,
    PzeudoOpsMul, PzeudoOpsSub, TensorTrait,
};

pub struct Tensor<'backward_label, F, GradStorage>
where
    GradStorage: GradientStorage<ArrayD<F>>,
{
    pub(crate) array: ArrayD<F>,
    pub(crate) gradient: Option<usize>,
    pub(crate) backward_label: Option<usize>,
    pub(crate) label_ops: AtomicBool,
    pub(crate) grad_storage: Rc<RefCell<GradStorage>>,
}

impl<'bacward_label, F, GradStorage> Tensor<'bacward_label, F, GradStorage>
where
    F: Clone + Zero + One + Float + Add<Output = F> + Copy,
    GradStorage: GradientStorage<ArrayD<F>>,
{
    pub fn new(
        array: ArrayD<F>,
        gradient: Option<ArrayD<F>>,
        backward_label: Option<BackwardLabel<'bacward_label, F>>,
        grad_storage: Rc<RefCell<GradStorage>>,
    ) -> Tensor<'bacward_label, F, GradStorage> {
        Self {
            array,
            gradient: gradient.map(|grad| grad_storage.borrow_mut().push_grad(grad)),
            backward_label: backward_label.map(|label| label),
            label_ops: AtomicBool::new(false),
            grad_storage,
        }
    }

    pub fn from_array(
        array: ArrayD<F>,
        grad_storage: Rc<RefCell<GradStorage>>,
    ) -> Tensor<'bacward_label, F, GradStorage> {
        let grad = ArrayD::<F>::zeros(array.shape());
        let idx = grad_storage.borrow_mut().push_grad(grad);
        Self {
            gradient: Some(idx),
            array,
            backward_label: None,
            label_ops: AtomicBool::new(false),
            grad_storage,
        }
    }
}

impl<'backward_label, F, GradStorage> TensorTrait<'backward_label, F, ArrayD<F>, GradStorage>
    for Tensor<'backward_label, F, GradStorage>
where
    F: Clone + One + Zero,
    GradStorage: GradientStorage<ArrayD<F>>,
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

    fn get_share_gradient(&self) -> Option<usize> {
        self.gradient
    }

    fn get_share_backward_label(&self) -> Option<&BackwardLabel<'backward_label, F>> {
        self.backward_label.as_ref()
    }

    fn get_storage(&self) -> &Rc<RefCell<GradStorage>> {
        &self.grad_storage
    }

    fn set_gradient_ones(&self) -> Result<(), &str> {
        if let Some(grad) = &self.gradient {
            *self
                .grad_storage
                .borrow_mut()
                .get_mut_storage()
                .get_mut(*grad)
                .unwrap() = Some(ArrayD::<F>::ones(self.array.shape()));
            return Ok(());
        } else {
            return Err("Gradient Disable");
        }
    }
}

impl<'bacward_label, F, GradStorage: GradientStorage<ArrayD<F>>>
    PzeudoOpsAdd<'bacward_label, F, ArrayD<F>, GradStorage>
    for Tensor<'bacward_label, F, GradStorage>
where
    F: Add<Output = F> + Copy + Zero + Float,
{
}

impl<'bacward_label, F, GradStorage: GradientStorage<ArrayD<F>>>
    PzeudoOpsSub<'bacward_label, F, ArrayD<F>, GradStorage>
    for Tensor<'bacward_label, F, GradStorage>
where
    F: Add<Output = F> + Copy + Zero + Float,
{
}

impl<'bacward_label, F, GradStorage: GradientStorage<ArrayD<F>>>
    PzeudoOpsMul<'bacward_label, F, ArrayD<F>, GradStorage>
    for Tensor<'bacward_label, F, GradStorage>
where
    F: Add<Output = F> + Copy + Zero + Float,
{
}

impl<'bacward_label, F, GradStorage: GradientStorage<ArrayD<F>>>
    PzeudoOpsDiv<'bacward_label, F, ArrayD<F>, GradStorage>
    for Tensor<'bacward_label, F, GradStorage>
where
    F: Add<Output = F> + Copy + Zero + Float,
{
}

// impl<'bacward_label, F> PzeudoOpsMatmul2D<'bacward_label, F> for Tensor<'bacward_label, F> where
//     F: Add<Output = F> + Copy + Zero + Float
// {
// }

// impl<'bacward_label, F> Backward<'bacward_label, F> for Tensor<'bacward_label, F>
// where
//     F: AddAssign<F>
//         + Clone
//         + Copy
//         + Zero
//         + Neg<Output = F>
//         + Mul<Output = F>
//         + Div<Output = F>
//         + One,
//     for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
// {
// }

// impl<'bacward_label, F> Display for Tensor<'bacward_label, F>
// where
//     F: Display,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.write_str(&format!("{}", self.array))
//     }
// }
