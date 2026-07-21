use pzeudo::{Array, ArrayTrait, OpsAdd, OpsMatmul2DF32, OpsPermute, OpsSlicing, OpsToShape, r};
use std::{assert_eq, println, vec};

fn main() {
    let shape = [2, 3, 4];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", array_a);

    let shape = [1, 3, 1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("{}", array_b);

    let check = vec_a.iter().map(|x| x + 0.10).collect::<Vec<f32>>();
    let result = array_a.add(&array_b).unwrap();
    // assert_eq!(check)
    println!("{}", result);
}
