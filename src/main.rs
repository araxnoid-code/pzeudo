use std::println;

use pzeudo::*;

fn main() {
    // let module = Module::<f32>::new();

    let shape = [3, 3, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", tensor_a);

    let data = tensor_a.powi(-1).unwrap();
    println!("{}", data);
}
