use std::fmt::{Debug, Display};

use pzeudo::{EpochBuilder, Grad, Linear, Module, NoGrad, Sgd, mse, r};

// fn main() {
//     let module = Module::<f32>::new();

//     let shape = [8, 1];
//     let vec = (0..shape.iter().product::<usize>())
//         .map(|x| x as f32 * 0.1)
//         .collect::<Vec<f32>>();

//     let mut tensor_a = module
//         .tensor_from_vector_with_shape::<Grad>(&vec, &shape)
//         .unwrap();

//     let mut tensor_b = module
//         .tensor_from_vector_with_shape::<Grad>(&vec, &shape)
//         .unwrap();

//     let mut tensor_c = tensor_a.add(&tensor_b, Grad).unwrap();

//     tensor_a.no_grad().unwrap();

//     tensor_c.backward().unwrap();

//     // dataset.backward().unwrap();
// }
