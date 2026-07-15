use crate::{Array, OpsDiv};

#[test]
fn div_test_1() {
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
        .div(&array_b)
        .map_or(Ok(()), |_| Err("Test 1.1 Error"))
        .unwrap();
}

#[test]
fn div_test_2() {
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

    let array_c = array_a / array_b;
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| *a / *b)
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

    let array_c = array_a / array_b;
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| *a / *b)
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

    let array_c = array_a / array_b;
    let check = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| *a / *b)
        .collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);
}

#[test]
fn div_test_3() {
    // Test 3. Contiguos Scalar Operation
    // // Test 3.1 1D
    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 10.;

    let array_c = array_a.div_scalar(scalar).unwrap();
    let check = vec_a.iter().map(|a| *a / scalar).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    let shape = [32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 20.;

    let array_c = array_a.scalar_div(scalar).unwrap();
    let check = vec_a.iter().map(|a| scalar / *a).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    // // Test 3.2 3D
    let shape = [3, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 25.;

    let array_c = array_a.div_scalar(scalar).unwrap();
    let check = vec_a.iter().map(|a| *a / scalar).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    let shape = [3, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 25.;

    let array_c = array_a.scalar_div(scalar).unwrap();
    let check = vec_a.iter().map(|a| scalar / *a).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    // // Test 3.3 5D
    let shape = [3, 3, 8, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 32.;

    let array_c = array_a.div_scalar(scalar).unwrap();
    let check = vec_a.iter().map(|a| *a / scalar).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);

    let shape = [3, 3, 8, 16, 16];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let scalar = 32.;

    let array_c = array_a.scalar_div(scalar).unwrap();
    let check = vec_a.iter().map(|a| scalar / *a).collect::<Vec<f32>>();
    assert_eq!(array_c.data, check);
}
