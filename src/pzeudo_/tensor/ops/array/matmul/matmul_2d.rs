use std::ops::AddAssign;

use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, Ix2, linalg::Dot};

use crate::{PzeudoOpsErr, StorageTrait};

pub fn matmul_2d<F>(lhs: ArrayViewD<F>, rhs: ArrayViewD<F>) -> Result<ArrayD<F>, PzeudoOpsErr>
where
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(PzeudoOpsErr::Matmul2dErr(format!(
            "Matmul2dErr. Cannot perform matmul_2d on an array of shape {lhs_shape:?} and an array of shape {rhs_shape:?} because they must be 2-dimensional."
        )));
    } else if lhs_shape[1] != rhs_shape[0] {
        return Err(PzeudoOpsErr::Matmul2dErr(format!(
            "Matmul2dErr. Cannot perform matmul_2d on an array of shape {lhs_shape:?} and an array of shape {rhs_shape:?} because it does not meet 'mxk * kxn'."
        )));
    }

    let lhs_2d = lhs.view().into_dimensionality::<Ix2>().map_err(|_| {
        PzeudoOpsErr::Matmul2dErr(format!(
            "Matmul2dErr. Cannot convert nd array with shape {lhs_shape:?} to 2d array"
        ))
    })?;
    let rhs_2d = rhs.view().into_dimensionality::<Ix2>().map_err(|_| {
        PzeudoOpsErr::Matmul2dErr(format!(
            "Matmul2dErr. Cannot convert nd array with shape {rhs_shape:?} to 2d array"
        ))
    })?;

    Ok(lhs_2d.dot(&rhs_2d).into_dyn())
}

pub fn matmul_2d_backward<F, GradStorage>(
    lhs: ArrayViewD<F>,
    lhs_grad_idx: Option<usize>,
    rhs: ArrayViewD<F>,
    rhs_grad_idx: Option<usize>,
    gradient_idx: Option<usize>,
    grad_storage: &mut GradStorage,
) -> Result<(), PzeudoOpsErr>
where
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
    F: AddAssign + Copy,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    let (lhs_grad, rhs_grad) = if let Some(grad_idx) = gradient_idx {
        let gradient = grad_storage
            .get_element(grad_idx)
            .map_err(|err| PzeudoOpsErr::Matmul2dBackwardErr(err.to_string()))?;
        let gradient_2d = gradient.view().into_dimensionality::<Ix2>().map_err(|_| {
            PzeudoOpsErr::Matmul2dErr(format!(
                "Matmul2dErr. Cannot convert nd array with shape {:?} to 2d array",
                gradient.shape()
            ))
        })?;

        let lhs_grad = if let Some(_) = lhs_grad_idx {
            let rhs_grad_2d = rhs.view().into_dimensionality::<Ix2>().map_err(|_| {
                PzeudoOpsErr::Matmul2dErr(format!(
                    "Matmul2dErr. Cannot convert nd array with shape {:?} to 2d array",
                    rhs.shape()
                ))
            })?;

            Some(rhs_grad_2d.t().dot(&gradient_2d.view()))
        } else {
            None
        };

        let rhs_grad = if let Some(_) = rhs_grad_idx {
            let lhs_grad_2d = lhs.view().into_dimensionality::<Ix2>().map_err(|_| {
                PzeudoOpsErr::Matmul2dErr(format!(
                    "Matmul2dErr. Cannot convert nd array with shape {:?} to 2d array",
                    lhs.shape()
                ))
            })?;
            Some(gradient_2d.dot(&lhs_grad_2d.t()))
        } else {
            None
        };

        (lhs_grad, rhs_grad)
    } else {
        (None, None)
    };

    if let (Some(assign), Some(grad_idx)) = (lhs_grad, lhs_grad_idx) {
        let grad = grad_storage.get_mut_element(grad_idx).unwrap();
        grad.add_assign(&assign);
    }

    if let (Some(assign), Some(grad_idx)) = (rhs_grad, rhs_grad_idx) {
        let grad = grad_storage.get_mut_element(grad_idx).unwrap();
        grad.add_assign(&assign);
    }

    Ok(())
}
