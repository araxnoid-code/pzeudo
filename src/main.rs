use pzeudo::{
    Array, ArrayStorage, ArrayTrait, OpsAdd, OpsMatmul2DF32, OpsPermute, OpsSlicing, OpsToShape,
    Tensor, TensorAddOps, TensorTrait, View, r,
};
use std::{assert_eq, cell::RefCell, println, rc::Rc, vec};

fn main() {
    let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
    let record = Rc::new(RefCell::new(Vec::new()));

    let shape = [4, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a =
        Tensor::from_vector_with_shape(&vec_a, &shape, storage.clone(), record.clone()).unwrap();

    let shape = [1, 2, 1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_b =
        Tensor::from_vector_with_shape(&vec_b, &shape, storage.clone(), record.clone()).unwrap();

    let result = tensor_a.add(&tensor_b).unwrap();

    result.backward().unwrap();

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<View>(tensor_a.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<View>(tensor_b.get_grad_idx().unwrap())
            .unwrap()
    )
}
