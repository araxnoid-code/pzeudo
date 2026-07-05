use std::{
    cell::RefCell,
    ops::{Add, Div, Neg},
    rc::Rc,
};

use ndarray::{Array2, ArrayBase, ArrayD, ArrayView2, ArrayViewD, Ix1, array, linalg::Dot};
use num_traits::{Float, One};
use pzeudo::{
    Backward, GradStorage, PzeudoOpsAdd, PzeudoOpsDiv, PzeudoOpsMatmul2D, Tensor, TensorView,
    matmul_2d,
};

fn main() {
    let data = vec![10.];
    let grad_storage = Rc::new(RefCell::new(GradStorage::new(None)));

    let array_a = ArrayD::<f32>::zeros(vec![5, 10]);
    let array_b = ArrayD::<f32>::zeros(vec![10, 1]);

    let tensor_a = Tensor::from_array(array_a, grad_storage.clone());
    let tensor_b = Tensor::from_array(array_b, grad_storage);

    let mut record = vec![];
    tensor_a.add(&tensor_b, &mut record);

    // tensor_b

    // tensor_a.add(&tensor_b, &mut record);

    // matmul_2d(array_a.view(), array_b.view()).unwrap();
}
