use pzeudo::*;

fn main() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [1, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();
    println!("{}", tensor_a);

    let shape = [3, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();
    println!("{}", tensor_b);

    let shape = [6, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();
    println!("{}", tensor_c);

    let shape = [2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();
    println!("{}", tensor_d);

    let concat = vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(0, Grad)
        .unwrap();
    println!("{}", concat);
    concat
        .value_vec_eq(&[
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0,
        ])
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(1, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();
}
