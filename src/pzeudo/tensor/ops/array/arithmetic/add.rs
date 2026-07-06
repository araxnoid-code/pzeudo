use std::{
    cell::RefCell,
    ops::{Add, AddAssign},
    rc::Rc,
};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Axis, Dim, IxDynImpl, OwnedRepr};
use num_traits::Zero;

use crate::{PzeudoOpsErr, able_broadcast};

pub fn add<F>(
    lhs: ArrayViewD<F>,
    rhs: ArrayViewD<F>,
) -> Result<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>, PzeudoOpsErr>
where
    F: Add<Output = F> + Copy,
{
    if lhs.shape().len() < rhs.shape().len() {
        able_broadcast(lhs.shape(), rhs.shape())?;
    } else {
        able_broadcast(rhs.shape(), lhs.shape())?;
    }

    Ok(&lhs + &rhs)
}

pub fn add_backward<F>(
    lhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
    rhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
    gradient: &Option<Rc<RefCell<ArrayD<F>>>>,
) where
    F: Clone + Zero + AddAssign<F>,
{
    // f(x, y) = x + y
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        // df(x, y)/dx = 1
        if let Some(lhs_grad) = lhs_grad {
            let mut lhs = lhs_grad.borrow_mut();
            if gradient.shape() == lhs.shape() {
                lhs.add_assign(&gradient.view());
            } else {
                let gradient_shape = gradient.shape();
                let mut lhs_shape = lhs.shape().to_vec();
                let distance = gradient_shape.len() - lhs_shape.len();
                if distance != 0 {
                    let ones = vec![1; distance];
                    lhs_shape = [ones, lhs_shape].concat();
                }

                let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> = None;
                for (idx, (gradient_shape, lhs_shape)) in
                    gradient_shape.iter().zip(lhs_shape.iter()).enumerate()
                {
                    if *lhs_shape == 1 && *gradient_shape != 1 {
                        if let Some(grad) = gradient_axis {
                            gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
                        } else {
                            gradient_axis =
                                Some(gradient.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
                        }
                    }
                }

                let gradient = gradient_axis.unwrap();
                let gradient = gradient.to_shape(lhs.shape()).unwrap();
                lhs.add_assign(&gradient);
            }
        }

        // df(x, y)/dy = 1
        if let Some(rhs_grad) = rhs_grad {
            let mut rhs = rhs_grad.borrow_mut();
            if gradient.shape() == rhs.shape() {
                rhs.add_assign(&gradient.view());
            } else {
                let gradient_shape = gradient.shape();
                let mut rhs_shape = rhs.shape().to_vec();
                let distance = gradient_shape.len() - rhs_shape.len();
                if distance != 0 {
                    let ones = vec![1; distance];
                    rhs_shape = [ones, rhs_shape].concat();
                }

                let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> = None;
                for (idx, (gradient_shape, rhs_shape)) in
                    gradient_shape.iter().zip(rhs_shape.iter()).enumerate()
                {
                    if *rhs_shape == 1 && *gradient_shape != 1 {
                        if let Some(grad) = gradient_axis {
                            gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
                        } else {
                            gradient_axis =
                                Some(gradient.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
                        }
                    }
                }

                let gradient = gradient_axis.unwrap();
                let gradient = gradient.to_shape(rhs.shape()).unwrap();
                rhs.add_assign(&gradient);
            }
        }
    }
}
