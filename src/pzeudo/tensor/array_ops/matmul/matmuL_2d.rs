use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, Ix2, linalg::Dot, s};

use crate::PzeudoErr;

pub fn matmul_2d<F>(lhs: ArrayViewD<F>, rhs: ArrayViewD<F>) -> Result<ArrayD<F>, PzeudoErr>
where
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(PzeudoErr::Matmul2dErr(format!(
            "Matmul2dErr, cannot do 2d matmul on array of shape {:?} and array of shape {:?}, because it is not 2-dimensional",
            lhs_shape, rhs_shape
        )));
    }

    if lhs_shape.last().unwrap() != rhs_shape.first().unwrap() {
        return Err(PzeudoErr::Matmul2dErr(format!(
            "Matmul2dErr, cannot do 2d matmul on array of shape {:?} and array of shape {:?}, because the size does not meet the requirements. The size must be in the form of 'm x k' and 'k x n'",
            lhs_shape, rhs_shape
        )));
    }

    let lhs_matrix: ArrayView2<F> = lhs
        .into_dimensionality::<Ix2>()
        .map_err(|err| PzeudoErr::Matmul2dErr(format!("Matmul2dErr, {:?}", err)))?;

    let rhs_matrix: ArrayView2<F> = rhs
        .into_dimensionality::<Ix2>()
        .map_err(|err| PzeudoErr::Matmul2dErr(format!("Matmul2dErr, {:?}", err)))?;

    Ok(lhs_matrix.dot(&rhs_matrix).into_dyn())
}
