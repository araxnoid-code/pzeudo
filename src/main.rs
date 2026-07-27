use std::println;

use pzeudo::*;

fn main() {
    let module = Module::<f32>::new();

    let shape = [3, 1];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let tensor_a = module
        .tensor_from_vector_with_shape(&vec_a, &shape)
        .unwrap();
    println!("tensor a");
    println!("{}", tensor_a);

    println!("{}", size_of_val(&tensor_a));

    // let broadcasted_a = tensor_a.broadcast(&[3, 3]).unwrap();
    // println!("slice a\n{broadcasted_a}");

    // // broadcasted_a.backward().unwrap();

    // let shape = [3, 3];
    // let vec_b = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32)
    //     .collect::<Vec<f32>>();
    // let tensor_b = module
    //     .tensor_from_vector_with_shape(&vec_b, &shape)
    //     .unwrap();
    // println!("tensor b\n{}", tensor_b);

    // let tensor_c = broadcasted_a.mul(&tensor_b).unwrap();
    // println!("tensor c\n{}", tensor_c);

    // tensor_c.backward().unwrap();

    // println!(
    //     "grad\n{}",
    //     module
    //         .get_storage()
    //         .borrow()
    //         .get_as_array_ref::<View>(tensor_a.get_grad_idx().unwrap(), ContiguousType::Grad)
    //         .unwrap()
    // );
}
