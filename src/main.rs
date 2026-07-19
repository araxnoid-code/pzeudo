use std::{cell::RefCell, rc::Rc};

use pzeudo::{
    Array, ArrayRef, ArrayStorage, ArrayTrait, Contiguous, OpsAdd, OpsAddAssign, Tensor,
    TensorAddOps, TensorDivOps, TensorMulOps, TensorSubOps, TensorTrait,
};

fn main() {
    let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
    let record = Rc::new(RefCell::new(Vec::new()));

    let shape = [3, 4];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a =
        Tensor::from_vector_with_shape(&vec_a, &shape, storage.clone(), record.clone()).unwrap();
    println!("{}", tensor_a);

    let shape = [3, 4];
    let vec_b = (10..10 + shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_b =
        Tensor::from_vector_with_shape(&vec_b, &shape, storage.clone(), record.clone()).unwrap();
    println!("{}", tensor_b);

    let result_c = tensor_a.mul(&tensor_b).unwrap();
    println!("{}", result_c);
    result_c.backward().unwrap();

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(tensor_a.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(tensor_b.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_c.get_grad_idx().unwrap())
            .unwrap()
    );
}
