use std::ops::Add;

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayView, Dim, IxDynImpl, OwnedRepr, ViewRepr, linalg::Dot,
};
use pzeudo::{F64Base, NDarrayBackend, PzeudoBackend, Tensor};

fn main() {
    let array_a: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64> = ArrayD::<f64>::zeros(vec![2, 2]);
    let array_b: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64> = ArrayD::<f64>::zeros(vec![2, 2]);

    let tensor_inner = Tensor::new(NDarrayBackend::new(F64Base::new(array_a)));
}
