use crate::prelude::*;

#[test]
fn matmul_2d_test_1() {
    // TEST 1. not 2-dimensional
    // TEST 1.1
    let shape = [8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.1 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();
    // TEST 1.2
    let shape = [2, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.2 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();
    // TEST 1.3
    let shape = [8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.3 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();
    // TEST 1.4
    let shape = [2, 3, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [2, 8, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.4 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();

    // TEST 2 m×k x k×n error
    let shape = [8, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [3, 8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.1 ERROR. NOT `m×k x k×n`"))
        .unwrap();
}

#[test]
fn matmul_2d_test_2() {
    let shape = [16, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8, 12];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.matmul_2d(&array_b).unwrap();

    let mut check = vec![0.; 16 * 12];
    unsafe {
        matrixmultiply::sgemm(
            16,
            8,
            12,
            1.,
            vec_a.as_ptr(),
            8,
            1,
            vec_b.as_ptr(),
            12,
            1,
            0.,
            check.as_mut_ptr(),
            12,
            1,
        )
    };
    assert_eq!(check, result.data);
}
