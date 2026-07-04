use std::ops::{Add, Div, Neg};

use ndarray::{Array2, ArrayBase, ArrayD, ArrayView2, ArrayViewD, Ix1, array, linalg::Dot};
use num_traits::{Float, One};
use pzeudo::{
    Backward, PzeudoOpsAdd, PzeudoOpsDiv, PzeudoOpsMatmul2D, Tensor, TensorView, matmul_2d,
};

fn main() {
    let array_a = ArrayD::<f32>::zeros(vec![5, 10]);
    let array_b = ArrayD::<f32>::zeros(vec![10, 1]);

    let tensor_a = Tensor::from_array(array_a);
    let tensor_b = Tensor::from_array(array_b);

    let mut record = vec![];
    tensor_a.matmul_2d(&tensor_b, &mut record).unwrap();

    // matmul_2d(array_a.view(), array_b.view()).unwrap();
}

fn dot<'a, F>(lhs: &'a ArrayView2<F>, rhs: &'a ArrayView2<F>) -> Array2<F>
where
    F: 'a,
    ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
{
    lhs.dot(&rhs)
}
