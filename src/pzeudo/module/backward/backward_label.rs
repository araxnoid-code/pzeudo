use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, ArrayView, ArrayViewD};

pub enum BackwardLabel<'a> {
    Mul(
        ArrayViewD<'a, f32>,
        ArrayViewD<'a, f32>,
        Option<Rc<RefCell<ArrayD<f32>>>>,
        Option<Rc<RefCell<ArrayD<f32>>>>,
    ),
}
