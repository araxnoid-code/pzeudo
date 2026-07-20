use std::{assert_eq, vec};

use crate::prelude::*;

#[test]
fn matmul_2d_test_1_f32() {
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
fn matmul_2d_test_1_f64() {
    // TEST 1. not 2-dimensional
    // TEST 1.1
    let shape = [8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.1 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();
    // TEST 1.2
    let shape = [2, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.2 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();
    // TEST 1.3
    let shape = [8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.3 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();
    // TEST 1.4
    let shape = [2, 3, 8];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [2, 8, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.4 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();

    // TEST 2 m×k x k×n error
    let shape = [8, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [3, 8];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_2d(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.1 ERROR. NOT `m×k x k×n`"))
        .unwrap();
}

#[test]
fn matmul_2d_test_2() {
    // TEST 1
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

    // TEST 2
    assert_eq!(check, result.data);
    let shape = [12, 7];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [7, 29];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.matmul_2d(&array_b).unwrap();

    let mut check = vec![0.; 12 * 29];
    unsafe {
        matrixmultiply::dgemm(
            12,
            7,
            29,
            1.,
            vec_a.as_ptr(),
            7,
            1,
            vec_b.as_ptr(),
            29,
            1,
            0.,
            check.as_mut_ptr(),
            29,
            1,
        )
    };
    assert_eq!(check, result.data);
}

#[test]
fn matmul_2d_test_3() {
    // TEST 1
    let shape = [4, 16, 32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let permute = array_a.permute(&[2, 1, 0]).unwrap(); // [32, 16, 4]
    let slice = permute.slicing(&[r(16..24), r(7..15), r(..)]).unwrap();
    let view_a = slice.index(&[8]).unwrap();
    println!("{}", view_a);

    let shape = [32, 64];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    let reshape = array_b.reshape(&[32, 8, -1]).unwrap(); // [32, 4, 16]
    let t = reshape.t(); // [16, 4, 32]
    let permute = t.permute(&[0, 2, 1]).unwrap();
    let index = permute.index(&[5]).unwrap();
    let view_b = index.slicing(&[r(10..14), r(2..)]).unwrap();
    println!("{}", view_b);

    let result = view_a.matmul_2d(&view_b).unwrap();
    // Check
    let c_array_a = view_a.into_array().unwrap();
    let c_array_b = view_b.into_array().unwrap();
    let mut check = vec![0.; 8 * 6];
    unsafe {
        matrixmultiply::sgemm(
            8,
            4,
            6,
            1.,
            c_array_a.data.as_ptr(),
            4,
            1,
            c_array_b.data.as_ptr(),
            6,
            1,
            0.,
            check.as_mut_ptr(),
            6,
            1,
        );
    }
    assert_eq!(check, result.data);

    // TEST 2
    let shape = [4, 16, 32];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();
    let permute = array_a.permute(&[2, 1, 0]).unwrap(); // [32, 16, 4]
    let slice = permute.slicing(&[r(16..24), r(7..15), r(..)]).unwrap();
    let view_a = slice.index(&[8]).unwrap();
    println!("{}", view_a);

    let shape = [32, 64];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    let reshape = array_b.reshape(&[32, 8, -1]).unwrap(); // [32, 4, 16]
    let t = reshape.t(); // [16, 4, 32]
    let permute = t.permute(&[0, 2, 1]).unwrap();
    let index = permute.index(&[5]).unwrap();
    let view_b = index.slicing(&[r(10..14), r(2..)]).unwrap();
    println!("{}", view_b);

    let result = view_a.matmul_2d(&view_b).unwrap();
    // Check
    let c_array_a = view_a.into_array().unwrap();
    let c_array_b = view_b.into_array().unwrap();
    let mut check = vec![0.; 8 * 6];
    unsafe {
        matrixmultiply::dgemm(
            8,
            4,
            6,
            1.,
            c_array_a.data.as_ptr(),
            4,
            1,
            c_array_b.data.as_ptr(),
            6,
            1,
            0.,
            check.as_mut_ptr(),
            6,
            1,
        );
    }
    assert_eq!(check, result.data);
}
