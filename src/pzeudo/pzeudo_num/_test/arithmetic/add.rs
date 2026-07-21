use crate::prelude::*;
use std::assert_eq;

#[test]
fn add_test_1() {
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
        .add(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.1 Error"))
        .unwrap();
}

#[test]
fn add_test_2() {
    // Test 2. Normal Contiguous operation
    // // Test 2.1 1D
    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [32];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let array_c = array_a + array_b;
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| *a + *b)
        .collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    // // Test 2.2 3D
    let shape = [32, 16, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [32, 16, 8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let array_c = array_a + array_b;
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| *a + *b)
        .collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    // // Test 2.3 5D
    let shape = [32, 24, 8, 16, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [32, 24, 8, 16, 8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let array_c = array_a + array_b;
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| *a + *b)
        .collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);
}

#[test]
fn add_test_3() {
    // Test 3. Contiguos Scalar Operation
    // // Test 3.1 1D
    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 20.;

    let array_c = array_a.add_scalar(scalar).unwrap();
    let check = vec_a.iter().map(|a| *a + scalar).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 20.;

    let array_c = array_a.scalar_add(scalar).unwrap();
    let check = vec_a.iter().map(|a| scalar + *a).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    // // Test 3.2 3D
    let shape = [3, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 25.;

    let array_c = array_a.add_scalar(scalar).unwrap();
    let check = vec_a.iter().map(|a| *a + scalar).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    let shape = [3, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 25.;

    let array_c = array_a.scalar_add(scalar).unwrap();
    let check = vec_a.iter().map(|a| scalar + *a).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    // // Test 3.3 5D
    let shape = [3, 3, 8, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 32.;

    let array_c = array_a.add_scalar(scalar).unwrap();
    let check = vec_a.iter().map(|a| *a + scalar).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    let shape = [3, 3, 8, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 32.;

    let array_c = array_a.scalar_add(scalar).unwrap();
    let check = vec_a.iter().map(|a| scalar + *a).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);
}

#[test]
fn add_test_4() {
    // AutoBroadcasting
    // TEST 1
    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", array_a);

    let shape = [1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("{}", array_b);

    let check = vec_a.iter().map(|x| x + 0.10).collect::<Vec<f32>>();
    let result = array_a.add(&array_b).unwrap();
    assert_eq!(check, result.data);

    // TEST 2
    let shape = [5, 32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", array_a);

    let shape = [1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.11)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("{}", array_b);

    let check = vec_a.iter().map(|x| x + 0.11).collect::<Vec<f32>>();
    let result = array_a.add(&array_b).unwrap();
    assert_eq!(check, result.data);

    // TEST 3
    let shape = [5, 32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", array_a);

    let shape = [1, 1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.12)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("{}", array_b);

    let check = vec_a.iter().map(|x| x + 0.12).collect::<Vec<f32>>();
    let result = array_a.add(&array_b).unwrap();
    assert_eq!(check, result.data);

    // TEST 4
    let shape = [2, 3, 4];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", array_a);

    let shape = [1, 3, 1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.5)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("{}", array_b);

    let result = array_a.add(&array_b).unwrap();
    let check_vec = array_b
        .broadcast(&array_a.shape)
        .unwrap()
        .into_array()
        .unwrap()
        .data;

    let check = vec_a
        .iter()
        .zip(check_vec.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.data);

    // TEST 5
    let shape = [8, 2, 5, 6];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    println!("{}", array_a);

    let shape = [8, 1, 5, 1];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    println!("{}", array_b);

    let result = array_a.add(&array_b).unwrap();
    let check_vec = array_b
        .broadcast(&array_a.shape)
        .unwrap()
        .into_array()
        .unwrap()
        .data;

    let check = vec_a
        .iter()
        .zip(check_vec.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.data);
}
