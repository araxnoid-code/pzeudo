use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, Ix2, linalg::Dot};

use crate::PzeudoOpsErr;

pub fn matmul_2d<F>(
    lhs: ArrayViewD<'_, F>,
    rhs: ArrayViewD<'_, F>,
) -> Result<ArrayD<F>, PzeudoOpsErr>
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

    let lhs_2d = lhs.into_dimensionality::<Ix2>().map_err(|_| {
        PzeudoOpsErr::Matmul2dErr(format!("Matmul2dErr. Cannot convert nd array to 2d array"))
    })?;
    let rhs_2d = rhs.into_dimensionality::<Ix2>().map_err(|_| {
        PzeudoOpsErr::Matmul2dErr(format!("Matmul2dErr. Cannot convert nd array to 2d array"))
    })?;

    Ok(lhs_2d.dot(&rhs_2d).into_dyn())
}
