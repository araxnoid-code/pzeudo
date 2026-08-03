use std::assert_eq;

use crate::prelude::*;

#[test]

fn permute_test_1() {
    let module = Module::<f32>::new();

    let shape = [3, 3, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a = module
        .tensor_from_vector_with_shape::<Grad>(&vec_a, &shape)
        .unwrap();
    println!("tensor a");
    println!("{}", tensor_a);

    let permute_a = tensor_a.permute(&[0, 2, 1]).unwrap();
    println!("slice a\n{permute_a}");

    let shape = [3, 2, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_b = module
        .tensor_from_vector_with_shape::<Grad>(&vec_b, &shape)
        .unwrap();
    println!("tensor b\n{}", tensor_b);

    let tensor_c = permute_a.mul(&tensor_b, Grad).unwrap();
    println!("tensor c\n{}", tensor_c);

    tensor_c.backward().unwrap();

    let grad_a = module
        .get_storage()
        .borrow()
        .get_as_array_ref::<View>(tensor_a.get_grad_idx().unwrap(), ContiguousType::Grad)
        .unwrap()
        .into_array()
        .unwrap()
        .data
        .to_vec();

    let check_tensor = tensor_b.permute(&[0, 2, 1]).unwrap();
    let check_array = module
        .get_storage()
        .borrow()
        .get_as_array_ref::<View>(check_tensor.get_array_idx(), ContiguousType::Arr)
        .unwrap()
        .into_array()
        .unwrap()
        .data
        .to_vec();

    assert_eq!(check_array, grad_a);

    // assert_eq!()
}
