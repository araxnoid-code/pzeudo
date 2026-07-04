use std::{cell::RefCell, ops::AddAssign, rc::Rc};

use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, Ix2, linalg::Dot, s};

use crate::PzeudoErr;

pub fn matmul_2d<F>(lhs: ArrayViewD<F>, rhs: ArrayViewD<F>) -> Result<ArrayD<F>, PzeudoErr>
where
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(PzeudoErr::Matmul2DErr(format!(
            "Matmul2dErr, cannot do matmul_2d on array of shape {:?} and array of shape {:?}, because it is not 2-dimensional",
            lhs_shape, rhs_shape
        )));
    }

    if lhs_shape.last().unwrap() != rhs_shape.first().unwrap() {
        return Err(PzeudoErr::Matmul2DErr(format!(
            "Matmul2dErr, cannot do matmul_2d on array of shape {:?} and array of shape {:?}, because the size does not meet the requirements. The size must be in the form of 'm x k' and 'k x n'",
            lhs_shape, rhs_shape
        )));
    }

    let lhs_matrix: ArrayView2<F> = lhs.into_dimensionality::<Ix2>().map_err(|err| {
        PzeudoErr::Matmul2DErr(format!(
            "Matmul2dErr, into dimensionality problem on lhs, {:?}",
            err
        ))
    })?;

    let rhs_matrix: ArrayView2<F> = rhs.into_dimensionality::<Ix2>().map_err(|err| {
        PzeudoErr::Matmul2DErr(format!(
            "Matmul2dErr, into dimensionality problem on rhs, {:?}",
            err
        ))
    })?;

    Ok(lhs_matrix.dot(&rhs_matrix).into_dyn())
}

pub(crate) fn matmul_2d_backward<F>(
    lhs: ArrayViewD<F>,
    lhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
    rhs: ArrayViewD<F>,
    rhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
    gradient: &Option<Rc<RefCell<ArrayD<F>>>>,
) -> Result<(), PzeudoErr>
where
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
    F: AddAssign + Copy,
{
    if lhs.shape().len() != 2 || rhs.shape().len() != 2 {
        return Err(PzeudoErr::Matmul2DBackwardErr(format!(
            "BackwardMatmul2dErr, cannot perform matmul_2d_backward on an array with shape {:?} of lhs and an array with shape {:?} of rhs, because those arrays are not 2-dimensional arrays.",
            lhs.shape().len(),
            rhs.shape().len()
        )));
    }

    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        if gradient.shape().len() != 2 {
            return Err(PzeudoErr::Matmul2DBackwardErr(format!(
                "BackwardMatmul2dErr, cannot do matmul_2d_backward because gradient is {:?} dimensional, gradient must be 2 dimensional.",
                gradient.shape().len()
            )));
        }

        let gradient_matrix = gradient
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|err| {
                PzeudoErr::Matmul2DBackwardErr(format!(
                    "BackwardMatmul2dErr, into dimensionality problem on gradients, {:?}",
                    err
                ))
            })?;

        // lhs
        // df(lhs, rhs)/dlhs = gradient * rhs^T
        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_grad = lhs_grad.borrow_mut();
            let rhs_matrix = rhs.view().into_dimensionality::<Ix2>().map_err(|err| {
                PzeudoErr::Matmul2DBackwardErr(format!(
                    "BackwardMatmul2dErr, into dimensionality problem on rhs, {:?}",
                    err
                ))
            })?;

            if lhs_grad.shape().len() != 2 {
                return Err(PzeudoErr::Matmul2DBackwardErr(format!(
                    "BackwardMatmul2dErr, cannot do matmul_2d_backward because lhs_grad is {:?} dimensional, gradient must be 2 dimensional.",
                    gradient.shape().len()
                )));
            }

            let d_lhs = gradient_matrix.dot(&(rhs_matrix.view().t()));
            lhs_grad.add_assign(&d_lhs);
        }

        // rhs
        // df(lhs, rhs)/drhs = gradient * rhs^T
        if let Some(rhs_grad) = rhs_grad {
            let mut rhs_grad = rhs_grad.borrow_mut();
            let lhs_matrix = lhs.view().into_dimensionality::<Ix2>().map_err(|err| {
                PzeudoErr::Matmul2DBackwardErr(format!(
                    "BackwardMatmul2dErr, into dimensionality problem on rhs, {:?}",
                    err
                ))
            })?;

            if rhs_grad.shape().len() != 2 {
                return Err(PzeudoErr::Matmul2DBackwardErr(format!(
                    "BackwardMatmul2dErr, cannot do matmul_2d_backward because rhs_grad is {:?} dimensional, gradient must be 2 dimensional.",
                    gradient.shape().len()
                )));
            }

            let d_rhs = lhs_matrix.view().t().dot(&gradient_matrix);
            rhs_grad.add_assign(&d_rhs);
        }
    }

    Ok(())
}
