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

#[test]
fn concat_test_4() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [3, 2, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 3, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 4, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 1, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let concat = vec![
        &tensor_a.view().unwrap(),
        &tensor_b.view().unwrap(),
        &tensor_c.view().unwrap(),
        &tensor_d.view().unwrap(),
    ]
    .tensor_concat(1, Grad)
    .unwrap();

    let shape = concat.get_shape().to_vec();

    let vec = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let other_tensor = Tensor::from_vector_with_shape(&vec, &shape, &module_builder, Grad).unwrap();

    let mul = other_tensor.mul(&concat, Grad).unwrap();

    mul.backward().unwrap();

    tensor_a
        .grad_vec_eq(&[
            0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0, 40.0, 41.0, 42.0, 43.0,
        ])
        .unwrap();
    tensor_b
        .grad_vec_eq(&[
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0,
        ])
        .unwrap();
    tensor_c
        .grad_vec_eq(&[
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            36.0, 37.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
        ])
        .unwrap();
    tensor_d
        .grad_vec_eq(&[18.0, 19.0, 38.0, 39.0, 58.0, 59.0])
        .unwrap();
}

#[test]
fn concat_test_5() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [3, 2, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 3, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 4, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 1, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let concat = vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(1, Grad)
        .unwrap();

    let shape = concat.get_shape().to_vec();

    let vec = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let other_tensor = Tensor::from_vector_with_shape(&vec, &shape, &module_builder, Grad).unwrap();

    tensor_b.no_grad().unwrap();
    let mul = other_tensor.mul(&concat, Grad).unwrap();

    mul.backward().unwrap();
    tensor_a
        .grad_vec_eq(&[
            0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0, 40.0, 41.0, 42.0, 43.0,
        ])
        .unwrap();
    tensor_c
        .grad_vec_eq(&[
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            36.0, 37.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
        ])
        .unwrap();
    tensor_d
        .grad_vec_eq(&[18.0, 19.0, 38.0, 39.0, 58.0, 59.0])
        .unwrap();
}
