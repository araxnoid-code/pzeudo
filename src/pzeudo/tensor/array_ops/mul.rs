use std::{cell::RefCell, ops::AddAssign, rc::Rc};

use ndarray::{ArrayBase, ArrayD, ArrayView, ArrayViewD, Dim, IxDynImpl, OwnedRepr};

pub fn mul(
    lhs: ArrayViewD<f32>,
    rhs: ArrayViewD<f32>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
    &lhs * &rhs
}

pub fn mul_backward(
    lhs: ArrayViewD<f32>,
    lhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
    rhs: ArrayViewD<f32>,
    rhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
    gradient: &Option<Rc<RefCell<ArrayD<f32>>>>,
) {
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        if let Some(lhs_grad) = lhs_grad {
            lhs_grad.borrow_mut().add_assign(&(&rhs * &gradient.view()));
        }

        if let Some(rhs_grad) = rhs_grad {
            rhs_grad.borrow_mut().add_assign(&(&lhs * &gradient.view()));
        }
    }
}
