use crate::prelude::*;

#[test]
fn dot_product_test_1() {
    // Test 1. Check Err
    // // Test 1.1 Shape Not Same
    let shape = [10];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .dot(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.1 Error, Shape Not Same"))
        .unwrap();

    // // Test 1.1 not 1-dimension
    let shape = [10, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [10, 2];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .dot(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.2 Error, not 1-dimension"))
        .unwrap();

    // Test 2
    // // Test 2.1 Shape Not Same
    let shape = [10];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .dot_f32(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.1 Error, Shape Not Same"))
        .unwrap();

    // // Test 2.1 not 1-dimension
    let shape = [10, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [10, 2];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .dot_f32(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.2 Error, not 1-dimension"))
        .unwrap();

    // Test 3
    // // Test 3.1 Shape Not Same
    let shape = [10];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64 + 0.10)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .dot_f64(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.1 Error, Shape Not Same"))
        .unwrap();

    // // Test 3.1 not 1-dimension
    let shape = [10, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [10, 2];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64 + 0.10)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .dot_f64(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.2 Error, not 1-dimension"))
        .unwrap();
}

#[test]
fn dot_product_test_2() {
    // Test 1
    let shape = [256];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [256];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.dot(&array_b).unwrap();
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    assert_eq!(vec![1], result.shape);
    assert_eq!(check, result.data[0]);

    // Test 2
    let shape = [256];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [256];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.dot_f32(&array_b).unwrap();
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    assert_eq!(vec![1], result.shape);
    assert_eq!(check, result.data[0]);

    // Test 3
    let shape = [256];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [256];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64 + 0.10)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.dot_f64(&array_b).unwrap();
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    assert_eq!(vec![1], result.shape);
    assert_eq!(check, result.data[0]);
}
