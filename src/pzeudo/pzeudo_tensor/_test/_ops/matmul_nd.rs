use std::assert_eq;

use crate::prelude::*;

#[test]
fn tensor_matmul_nd_f32_test() {
    let module = Module::new();

    let shape = [2, 3, 4, 5];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = module
        .tensor_from_vector_with_shape(&vec_a, &shape)
        .unwrap();

    let shape = [2, 3, 5, 6];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 100.)
        .collect::<Vec<f32>>();
    let array_b = module
        .tensor_from_vector_with_shape(&vec_b, &shape)
        .unwrap();

    let tensor_c = array_a.matmul_nd(&array_b).unwrap();
    tensor_c.backward().unwrap();

    let storeage = module.get_storage().borrow();

    let grad_arr_a = storeage
        .get_as_array_ref::<Contiguous>(array_a.get_grad_idx().unwrap(), ContiguousType::Grad)
        .unwrap();

    let grad_arr_b = storeage
        .get_as_array_ref::<Contiguous>(array_b.get_grad_idx().unwrap(), ContiguousType::Grad)
        .unwrap();

    // CHECK
    let shape = [2, 3, 4, 6];
    let ones = Array::<f32>::ones(&shape);

    let array_b = storeage
        .get_as_array_ref::<Contiguous>(array_b.get_array_idx(), ContiguousType::Arr)
        .unwrap();

    let permute = array_b.permute(&[0, 1, 3, 2]).unwrap();
    let check = ones.matmul_nd(&permute).unwrap();
    assert_eq!(check.data, grad_arr_a.data);

    let array_a = storeage
        .get_as_array_ref::<Contiguous>(array_a.get_array_idx(), ContiguousType::Arr)
        .unwrap();

    let permute = array_a.permute(&[0, 1, 3, 2]).unwrap();
    let check = permute.matmul_nd(&ones).unwrap();
    assert_eq!(check.data, grad_arr_b.data);
}

#[test]
fn tensor_matmul_nd_f64_test() {
    let module = Module::new();

    let shape = [2, 3, 4, 5];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = module
        .tensor_from_vector_with_shape(&vec_a, &shape)
        .unwrap();

    let shape = [2, 3, 5, 6];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64 + 100.)
        .collect::<Vec<f64>>();
    let array_b = module
        .tensor_from_vector_with_shape(&vec_b, &shape)
        .unwrap();

    let tensor_c = array_a.matmul_nd(&array_b).unwrap();
    tensor_c.backward().unwrap();

    let storeage = module.get_storage().borrow();

    let grad_arr_a = storeage
        .get_as_array_ref::<Contiguous>(array_a.get_grad_idx().unwrap(), ContiguousType::Grad)
        .unwrap();

    let grad_arr_b = storeage
        .get_as_array_ref::<Contiguous>(array_b.get_grad_idx().unwrap(), ContiguousType::Grad)
        .unwrap();

    // CHECK
    let shape = [2, 3, 4, 6];
    let ones = Array::<f64>::ones(&shape);

    let array_b = storeage
        .get_as_array_ref::<Contiguous>(array_b.get_array_idx(), ContiguousType::Arr)
        .unwrap();

    let permute = array_b.permute(&[0, 1, 3, 2]).unwrap();
    let check = ones.matmul_nd(&permute).unwrap();
    assert_eq!(check.data, grad_arr_a.data);

    let array_a = storeage
        .get_as_array_ref::<Contiguous>(array_a.get_array_idx(), ContiguousType::Arr)
        .unwrap();

    let permute = array_a.permute(&[0, 1, 3, 2]).unwrap();
    let check = permute.matmul_nd(&ones).unwrap();
    assert_eq!(check.data, grad_arr_b.data);
}
