use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayView, ArrayViewD};

#[derive(Debug)]
pub enum BackwardLabel<'backward_label, F> {
    Add(
        (ArrayViewD<'backward_label, F>, Option<usize>),
        (ArrayViewD<'backward_label, F>, Option<usize>),
        Option<usize>,
    ),
    Sub(
        (ArrayViewD<'backward_label, F>, Option<usize>),
        (ArrayViewD<'backward_label, F>, Option<usize>),
        Option<usize>,
    ),
    Mul(
        (ArrayViewD<'backward_label, F>, Option<usize>),
        (ArrayViewD<'backward_label, F>, Option<usize>),
        Option<usize>,
    ),
    Div(
        (ArrayViewD<'backward_label, F>, Option<usize>),
        (ArrayViewD<'backward_label, F>, Option<usize>),
        Option<usize>,
    ),
    Matmul(
        (ArrayViewD<'backward_label, F>, Option<usize>),
        (ArrayViewD<'backward_label, F>, Option<usize>),
        Option<usize>,
    ),
}
