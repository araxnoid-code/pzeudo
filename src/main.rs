use std::{cell::RefCell, rc::Rc};

use pzeudo::{
    Array, ArrayStorage, ArrayTrait, Contiguous, OpsAdd, OpsAddAssign, Tensor, TensorAddOps,
    TensorDivOps, TensorMulOps, TensorSubOps,
};

fn main() {
    let mut array = Array::<f32>::zeros(&[2, 2, 2]);
    let number =
        Array::<f32>::from_vector_with_shape(&vec![1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2])
            .unwrap();

    array.add_assign(&number).unwrap();
    println!("{}", array);
    // let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
    // let record = Rc::new(RefCell::new(Vec::new()));

    // let shape = [32];
    // let vec_a = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32)
    //     .collect::<Vec<f32>>();
    // let tensor_a =
    //     Tensor::from_vector_with_shape(&vec_a, &shape, storage.clone(), record.clone()).unwrap();

    // // let result = TensorAddOps::add(&self, rhs);

    // let shape = [32];
    // let vec_b = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32)
    //     .collect::<Vec<f32>>();
    // let tensor_b =
    //     Tensor::from_vector_with_shape(&vec_b, &shape, storage.clone(), record.clone()).unwrap();

    // let view = tensor_b.view().unwrap();

    // let tensor_c = tensor_a.add(&view).unwrap().view().unwrap();

    // let tensor_d = tensor_c.add(&view);

    // let view = tensor_b.view().unwrap();

    // let tensor_c = view.div(&view).unwrap();

    // let multiple = tensor_c.mul(&view).unwrap().view().unwrap();

    // let tensor = tensor_a.sub(&multiple).unwrap();

    // let tensor = tensor_b.add(&tensor).unwrap();
}
