use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayView, ArrayViewD};

#[derive(Debug)]
pub enum BackwardLabel<'backward_label, F> {
    Add(
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        Option<Rc<RefCell<ArrayD<F>>>>,
    ),
    Sub(
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        Option<Rc<RefCell<ArrayD<F>>>>,
    ),
    Mul(
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        Option<Rc<RefCell<ArrayD<F>>>>,
    ),
    Div(
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        (
            ArrayViewD<'backward_label, F>,
            Option<Rc<RefCell<ArrayD<F>>>>,
        ),
        Option<Rc<RefCell<ArrayD<F>>>>,
    ),
}
