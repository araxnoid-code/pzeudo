use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{ArrayD, ArrayView, ArrayViewD};

pub enum BackwardLabel<'a> {
    Add(
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
    ),
    Sub(
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
    ),
    Mul(
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
    ),
    Div(
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
        (
            ArrayViewD<'a, f32>,
            Option<Rc<RefCell<ArrayD<f32>>>>,
            Option<Arc<BackwardLabel<'a>>>,
        ),
    ),
}
