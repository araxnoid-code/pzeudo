use std::{assert_eq, vec};

use crate::prelude::*;

#[test]
fn broadcast_test_1() {
    let module = Module::<f32>::new(42);

    let shape = [3, 1];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module, Grad).unwrap();

    let broadcasted_a = tensor_a.broadcast(&[3, 3]).unwrap();

    let shape = [3, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_b, &shape, &module, Grad).unwrap();

    let tensor_c = broadcasted_a.mul(&tensor_b, Grad).unwrap();

    tensor_c.backward().unwrap();

    let grad = module
        .get_storage()
        .borrow()
        .get_as_array_ref::<View>(tensor_a.get_grad_idx().unwrap(), ContiguousType::Grad)
        .unwrap()
        .data
        .to_vec();

    assert_eq!(vec![3., 12., 21.], grad);
}
