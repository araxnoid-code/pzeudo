use std::ops::{Add, Div, Neg};

use ndarray::{Array2, ArrayBase, ArrayD, ArrayView2, ArrayViewD, Ix1, array, linalg::Dot};
use num_traits::{Float, One};
use pzeudo::{Backward, PzeudoOpsAdd, PzeudoOpsDiv, Tensor, TensorView, matmul_2d};

fn main() {
    let array_a = ArrayD::<f32>::zeros(vec![5, 10, 3]);
    // let array_b = ArrayD::<f32>::zeros(vec![10, 1]);

    let transpose = array_a.t();

    println!("{:?}", transpose.shape());

    // matmul_2d(array_a.view(), array_b.view()).unwrap();
}

fn dot<'a, F>(lhs: &'a ArrayView2<F>, rhs: &'a ArrayView2<F>) -> Array2<F>
where
    F: 'a,
    ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
{
    lhs.dot(&rhs)
}
