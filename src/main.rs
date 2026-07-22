use pzeudo::{
    Array, ArrayStorage, ArrayTrait, Linear, Module, OpsAdd, OpsMatmul2DF32, OpsPermute,
    OpsSlicing, OpsToShape, Tensor, TensorTrait, View, r,
};
use std::{assert_eq, cell::RefCell, println, rc::Rc, vec};

fn main() {
    let module = Module::<f32>::new();
    let linear = Linear::new(3, 4, &module).unwrap();

    let shape = [8, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 1.)
        .collect::<Vec<f32>>();
    let tensor_a = module
        .tensor_from_vector_with_shape(&vec_a, &shape)
        .unwrap();

    // let view = tensor_a.view().unwrap();
    let tensor = linear.forward::<_>(&tensor_a).unwrap();
    tensor.backward().unwrap();
    println!("{}", tensor);

    // let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
    // let record = Rc::new(RefCell::new(Vec::new()));

    // let shape = [2, 3];
    // let vec_a = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32 + 1.)
    //     .collect::<Vec<f32>>();
    // let tensor_a =
    //     Tensor::from_vector_with_shape(&vec_a, &shape, storage.clone(), record.clone()).unwrap();
    // println!("{}", tensor_a);

    // let shape = [3, 2];
    // let vec_b = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32 + 7.)
    //     .collect::<Vec<f32>>();
    // let tensor_b =
    //     Tensor::from_vector_with_shape(&vec_b, &shape, storage.clone(), record.clone()).unwrap();
    // println!("{}", tensor_b);

    // let result = tensor_a.add::<_>(&tensor_b.view().unwrap()).unwrap();
    // println!("{}", result);

    // result.backward().unwrap();

    println!(
        "{}",
        module
            .get_storage()
            .borrow()
            .get_as_array_ref::<View>(
                linear.get_bias().get_grad_idx().unwrap(),
                pzeudo::ContiguousType::Arr
            )
            .unwrap()
    );

    println!(
        "{}",
        module
            .get_storage()
            .borrow()
            .get_as_array_ref::<View>(
                linear.get_bias().get_grad_idx().unwrap(),
                pzeudo::ContiguousType::Grad
            )
            .unwrap()
    );

    // println!(
    //     "{}",
    //     storage
    //         .borrow()
    //         .get_as_array_ref::<View>(tensor_b.get_grad_idx().unwrap())
    //         .unwrap()
    // );
}
