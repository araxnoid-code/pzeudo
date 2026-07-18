use std::{cell::RefCell, rc::Rc};

use pzeudo::{Array, ArrayStorage, ArrayTrait, Contiguous, Tensor, TensorAddOps};

fn main() {
    let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
    let record = Rc::new(RefCell::new(Vec::new()));

    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a =
        Tensor::from_vector_with_shape(&vec_a, &shape, storage.clone(), record.clone()).unwrap();

    // let result = TensorAddOps::add(&self, rhs);

    let shape = [32];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_b =
        Tensor::from_vector_with_shape(&vec_b, &shape, storage.clone(), record.clone()).unwrap();

    let view = tensor_b.view().unwrap();

    let tensor_c = tensor_a.add(&view).unwrap().view().unwrap();

    let tensor_d = tensor_c.add(&view);

    // // let view = tensor_b.view().unwrap();

    // let tensor_c = view.div(&view).unwrap();

    // let multiple = tensor_c.mul(&view).unwrap().view().unwrap();

    // let tensor = tensor_a.sub(&multiple).unwrap();
}
