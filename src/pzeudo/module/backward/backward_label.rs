use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayView, ArrayViewD};

#[derive(Debug)]
pub enum BackwardLabel<'backward_label> {
    Add(
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        Option<Rc<RefCell<ArrayD<f32>>>>,
    ),
    Sub(
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        Option<Rc<RefCell<ArrayD<f32>>>>,
    ),
    Mul(
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        Option<Rc<RefCell<ArrayD<f32>>>>,
    ),
    Div(
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        (
            ArrayViewD<'backward_label, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
        ),
        Option<Rc<RefCell<ArrayD<f32>>>>,
    ),
}
