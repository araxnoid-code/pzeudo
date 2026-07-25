use std::println;

use pzeudo::{Array, ArrayTrait, OpsMatmul2DF32, OpsMatmul2DF64, OpsMatmulNDF32, OpsMatmulNDF64};

fn main() {
    let shape = [3, 3, 3, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("array a");
    println!("{}", array_a);

    let shape = [3, 3, 3, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("array b");
    println!("{}", array_b);

    let result = array_a.matmul_nd(&array_b).unwrap();
    println!("result");
    println!("{}", result);

    println!("check");

    for i in 0..3 {
        for j in 0..3 {
            let index_array_a = array_a.index(&[i, j]).unwrap();
            let index_array_b = array_b.index(&[i, j]).unwrap();
            println!("{}", index_array_a.matmul_2d(&index_array_b).unwrap());
        }
    }
}
