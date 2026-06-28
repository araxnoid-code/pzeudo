use std::{cell::RefCell, ops::AddAssign, rc::Rc};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Dim, IxDynImpl, OwnedRepr};

pub fn div(
    lhs: ArrayViewD<f32>,
    rhs: ArrayViewD<f32>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
    &lhs / &rhs
}

pub fn div_backward(
    lhs: ArrayViewD<f32>,
    lhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
    rhs: ArrayViewD<f32>,
    rhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
    gradient: &Option<Rc<RefCell<ArrayD<f32>>>>,
) {
    // f(lhs, rhs) = lhs / rhs
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        if let Some(lhs_grad) = lhs_grad {
            // df(lhs, rhs)/dlhs = 1/rhs
            lhs_grad
                .borrow_mut()
                .add_assign(&(1. / &rhs * &gradient.view()));
        }

        if let Some(rhs_grad) = rhs_grad {
            // df(lhs, rhs)/drhs =  -lhs/rhs2
            rhs_grad
                .borrow_mut()
                .add_assign(&(-&lhs / rhs.pow2() * &gradient.view()));
        }
    }
}
