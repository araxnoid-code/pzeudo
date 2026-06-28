use std::{cell::RefCell, ops::AddAssign, rc::Rc};

use ndarray::{ArrayBase, ArrayD, ArrayView, ArrayViewD, Dim, IxDynImpl, OwnedRepr};

pub fn add(
    lhs: ArrayViewD<f32>,
    rhs: ArrayViewD<f32>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
    &lhs * &rhs
}

pub fn add_backward(
    lhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
    rhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
    gradient: &Option<Rc<RefCell<ArrayD<f32>>>>,
) {
    // f(x, y) = x + y
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        if let Some(lhs_grad) = lhs_grad {
            // df(x, y)/dx = 1
            lhs_grad.borrow_mut().add_assign(&gradient.view());
        }

        if let Some(rhs_grad) = rhs_grad {
            // df(x, y)/dy = 1
            rhs_grad.borrow_mut().add_assign(&gradient.view());
        }
    }
}
