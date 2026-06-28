use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, ArrayView, ArrayViewD};

pub enum BackwardLabel<'a> {
    Add(
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
    ),
    Sub(
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
    ),
    Mul(
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
    ),
    Div(
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
        (ArrayViewD<'a, f32>, Option<Rc<RefCell<ArrayD<f32>>>>),
    ),
}
