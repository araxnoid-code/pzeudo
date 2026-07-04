use std::{
    cell::RefCell,
    ops::{Add, AddAssign, Div, Mul, Neg},
    rc::Rc,
};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Axis, Dim, IxDynImpl, OwnedRepr};
use num_traits::{Float, One, Zero, one};

use crate::{PzeudoErr, able_broadcast, neg, pow2};

pub fn div<F>(
    lhs: ArrayViewD<F>,
    rhs: ArrayViewD<F>,
) -> Result<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>, PzeudoErr>
where
    F: Div<Output = F> + Copy + Float,
{
    if lhs.shape().len() < rhs.shape().len() {
        able_broadcast(lhs.shape(), rhs.shape())
            .map_err(|err| PzeudoErr::DivErr(err.into_msg()))?;
    } else {
        able_broadcast(rhs.shape(), lhs.shape()).map_err(|err| PzeudoErr::AddErr(err.into_msg()))?
    }

    Ok(&lhs / &rhs)
}

pub fn div_backward<F>(
    lhs: ArrayViewD<F>,
    lhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
    rhs: ArrayViewD<F>,
    rhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
    gradient: &Option<Rc<RefCell<ArrayD<F>>>>,
) where
    F: Div<Output = F>
        + Copy
        + Mul<Output = F>
        + Zero
        + AddAssign<F>
        + Clone
        + Neg<Output = F>
        + One,
{
    // f(lhs, rhs) = lhs / rhs
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        // df(lhs, rhs)/dlhs = 1/rhs
        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_grad = lhs_grad.borrow_mut();

            if gradient.shape() == lhs_grad.shape() {
                let assign = &scalar_div(one(), rhs.view()).view() * &gradient.view();
                lhs_grad.add_assign(&assign);
            } else {
                let gradient_shape = gradient.shape();
                let mut lhs_grad_shape = lhs_grad.shape().to_vec();
                let distance = gradient_shape.len() - lhs_grad_shape.len();
                if distance != 0 {
                    let ones = vec![1; distance];
                    lhs_grad_shape = [ones, lhs_grad_shape].concat();
                }

                let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> =
                    Some(scalar_div(one(), rhs.view()) * &gradient.view());

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
                rhs_grad.add_assign(&(neg(lhs) / pow2(rhs.view()) * &gradient.view()));
            } else {
                let gradient_shape = gradient.shape();
                let mut rhs_grad_shape = rhs_grad.shape().to_vec();
                let distance = gradient_shape.len() - rhs_grad_shape.len();
                if distance != 0 {
                    let ones = vec![1; distance];
                    rhs_grad_shape = [ones, rhs_grad_shape].concat();
                }

                // pow2(rhs.view()) * &gradient.view();
                let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> =
                    Some(neg(lhs) / pow2(rhs.view()) * &gradient.view());

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

fn scalar_div<F>(scalar: F, arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Div<Output = F> + Clone + Copy,
{
    arr.mapv(|x| scalar / x)
}
