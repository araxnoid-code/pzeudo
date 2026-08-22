use crate::prelude::*;

#[test]
fn concat_test_1() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [3, 4, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 1, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 3, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let concat = vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(1, Grad)
        .unwrap();
    //

    concat
        .value_vec_eq(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0,
        ])
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(0, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(2, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();
}

#[test]
fn concat_test_2() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [3, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [2, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [1, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [4, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let concat = vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(0, Grad)
        .unwrap();

    concat
        .value_vec_eq(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ])
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(1, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(2, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();
}

#[test]
fn concat_test_3() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [3, 2, 1];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 2, 3];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 2, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 2, 4];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let concat = vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(2, Grad)
        .unwrap();

    concat
        .value_vec_eq(&[
            0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0,
            3.0, 3.0, 3.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 0.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 0.0,
            1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
        ])
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(0, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();

    vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(1, Grad)
        .map_or(Ok(()), |_| Err("Error on a different test shape off-axis."))
        .unwrap();
}
