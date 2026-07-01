use std::{cell::RefCell, ops::AddAssign, rc::Rc};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Axis, Dim, IxDynImpl, OwnedRepr};

use crate::{PzeudoErr, able_broadcast};

pub fn div(
    lhs: ArrayViewD<f32>,
    rhs: ArrayViewD<f32>,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>, PzeudoErr> {
    if lhs.shape() < rhs.shape() {
        able_broadcast(lhs.shape(), rhs.shape())
            .map_err(|err| PzeudoErr::DivErr(err.into_msg()))?;
    } else {
        able_broadcast(rhs.shape(), lhs.shape()).map_err(|err| PzeudoErr::AddErr(err.into_msg()))?
    }

    Ok(&lhs / &rhs)
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

        // df(lhs, rhs)/dlhs = 1/rhs
        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_grad = lhs_grad.borrow_mut();

            if gradient.shape() == lhs_grad.shape() {
                lhs_grad.add_assign(&(1. / &rhs * &gradient.view()));
            } else {
                let gradient_shape = gradient.shape();
                let mut lhs_grad_shape = lhs_grad.shape().to_vec();
                let distance = gradient_shape.len() - lhs_grad_shape.len();
                if distance != 0 {
                    let ones = vec![1; distance];
                    lhs_grad_shape = [ones, lhs_grad_shape].concat();
                }

                let mut gradient_axis: Option<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>> =
                    Some(1. / &rhs * &gradient.view());

                for (idx, (gradient_shape, lhs_grad_shape)) in
                    gradient_shape.iter().zip(lhs_grad_shape.iter()).enumerate()
                {
                    if *lhs_grad_shape == 1 && *gradient_shape != 1 {
                        if let Some(grad) = gradient_axis {
                            gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
                        }
                    }
                }

                let gradient = gradient_axis.unwrap();
                let gradient = gradient.to_shape(lhs_grad.shape()).unwrap();
                lhs_grad.add_assign(&gradient);
            }
        }

        // df(lhs, rhs)/drhs =  -lhs/rhs^2
        if let Some(rhs_grad) = rhs_grad {
            let mut rhs_grad = rhs_grad.borrow_mut();

            if gradient.shape() == rhs_grad.shape() {
                rhs_grad.add_assign(&(-&lhs / rhs.pow2() * &gradient.view()));
            } else {
                let gradient_shape = gradient.shape();
                let mut rhs_grad_shape = rhs_grad.shape().to_vec();
                let distance = gradient_shape.len() - rhs_grad_shape.len();
                if distance != 0 {
                    let ones = vec![1; distance];
                    rhs_grad_shape = [ones, rhs_grad_shape].concat();
                }

                let mut gradient_axis: Option<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>> =
                    Some(-&lhs / rhs.pow2() * &gradient.view());

                for (idx, (gradient_shape, rhs_shape)) in
                    gradient_shape.iter().zip(rhs_grad_shape.iter()).enumerate()
                {
                    if *rhs_shape == 1 && *gradient_shape != 1 {
                        if let Some(grad) = gradient_axis {
                            gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
                        }
                    }
                }

                let gradient = gradient_axis.unwrap();
                let gradient = gradient.to_shape(rhs_grad.shape()).unwrap();
                rhs_grad.add_assign(&gradient);
            }
        }
    }
}
