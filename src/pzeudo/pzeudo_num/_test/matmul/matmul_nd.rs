use std::{arch::x86_64::_mm_maskz_srai_epi32, assert_eq, panic};

use crate::prelude::*;

#[test]
fn matmul_nd_test_1_f32() {
    // TEST 1. not 3 or 2-dimensional
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
        .matmul_nd(&array_b)
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
        .matmul_nd(&array_b)
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
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.3 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();

    // TEST 2. 3 dimensional but shape not same
    // TEST 2.1
    let shape = [2, 3, 4];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [1, 4, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| {
            Err("TEST 2.1 ERROR. 3 DIMENSIONAL BUT SHAPE NOT SAME")
        })
        .unwrap();

    // TEST 2.2
    let shape = [2, 4, 7];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [3, 2, 7, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| {
            Err("TEST 2.1 ERROR. 3 DIMENSIONAL BUT SHAPE NOT SAME")
        })
        .unwrap();

    // TEST 3. m×k x k×n error
    let shape = [2, 4, 7];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [2, 4, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| Err("TEST 3 ERROR. NOT `m×k x k×n`"))
        .unwrap();
}

#[test]
fn matmul_nd_test_1_f64() {
    // TEST 1. not 3 or 2-dimensional
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
        .matmul_nd(&array_b)
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
        .matmul_nd(&array_b)
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
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| Err("TEST 1.3 ERROR. NOT 2-DIMENSIONAL"))
        .unwrap();

    // TEST 2. 3 dimensional but shape not same
    // TEST 2.1
    let shape = [2, 3, 4];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [1, 4, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| {
            Err("TEST 2.1 ERROR. 3 DIMENSIONAL BUT SHAPE NOT SAME")
        })
        .unwrap();

    // TEST 2.2
    let shape = [2, 4, 7];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [3, 2, 7, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| {
            Err("TEST 2.1 ERROR. 3 DIMENSIONAL BUT SHAPE NOT SAME")
        })
        .unwrap();

    // TEST 3. m×k x k×n error
    let shape = [2, 4, 7];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [2, 4, 3];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    array_a
        .matmul_nd(&array_b)
        .map_or(Ok(()), |_| Err("TEST 3 ERROR. NOT `m×k x k×n`"))
        .unwrap();
}

#[test]
fn matmul_nd_test_2_f32() {
    // TEST 1
    let shape = [8, 24, 11];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8, 11, 19];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32 + 0.10)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    let result = array_a.matmul_nd(&array_b).unwrap();

    // index 1
    let index_array_a = array_a.index(&[0]).unwrap();
    let index_array_b = array_b.index(&[0]).unwrap();
    let check = index_array_a.matmul_nd(&index_array_b).unwrap();
    let slice_result = result.index(&[0]).unwrap().into_array().unwrap().data;
    assert_eq!(check.data, slice_result);

    // index 2
    let index_array_a = array_a.index(&[1]).unwrap();
    let index_array_b = array_b.index(&[1]).unwrap();
    let check = index_array_a.matmul_nd(&index_array_b).unwrap();
    let slice_result = result.index(&[1]).unwrap().into_array().unwrap().data;
    assert_eq!(check.data, slice_result);

    // index 8
    let index_array_a = array_a.index(&[7]).unwrap();
    let index_array_b = array_b.index(&[7]).unwrap();
    let check = index_array_a.matmul_nd(&index_array_b).unwrap();
    let slice_result = result.index(&[7]).unwrap().into_array().unwrap().data;
    assert_eq!(check.data, slice_result);

    // TEST 2
    let shape = [7, 4, 6, 12, 9];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [7, 4, 6, 9, 11];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.matmul_nd(&array_b).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let index_array_a = array_a.index(&[i, j, k]).unwrap();
                let index_array_b = array_b.index(&[i, j, k]).unwrap();
                let check = index_array_a.matmul_2d(&index_array_b).unwrap();
                let index = result.index(&[i, j, k]).unwrap().into_array().unwrap();
                assert_eq!(check.data, index.data);
            }
        }
    }
}

#[test]
fn matmul_nd_test_2_f64() {
    // TEST 1
    let shape = [8, 24, 11];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [8, 11, 19];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64 + 0.10)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();
    let result = array_a.matmul_nd(&array_b).unwrap();

    // index 1
    let index_array_a = array_a.index(&[0]).unwrap();
    let index_array_b = array_b.index(&[0]).unwrap();
    let check = index_array_a.matmul_nd(&index_array_b).unwrap();
    let slice_result = result.index(&[0]).unwrap().into_array().unwrap().data;
    assert_eq!(check.data, slice_result);

    // index 2
    let index_array_a = array_a.index(&[1]).unwrap();
    let index_array_b = array_b.index(&[1]).unwrap();
    let check = index_array_a.matmul_nd(&index_array_b).unwrap();
    let slice_result = result.index(&[1]).unwrap().into_array().unwrap().data;
    assert_eq!(check.data, slice_result);

    // index 8
    let index_array_a = array_a.index(&[7]).unwrap();
    let index_array_b = array_b.index(&[7]).unwrap();
    let check = index_array_a.matmul_nd(&index_array_b).unwrap();
    let slice_result = result.index(&[7]).unwrap().into_array().unwrap().data;
    assert_eq!(check.data, slice_result);

    // TEST 2
    let shape = [7, 4, 6, 12, 9];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Array::from_vector_with_shape(&vec_a, &shape).unwrap();

    let shape = [7, 4, 6, 9, 11];
    let vec_b = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Array::from_vector_with_shape(&vec_b, &shape).unwrap();

    let result = array_a.matmul_nd(&array_b).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let index_array_a = array_a.index(&[i, j, k]).unwrap();
                let index_array_b = array_b.index(&[i, j, k]).unwrap();
                let check = index_array_a.matmul_2d(&index_array_b).unwrap();
                let index = result.index(&[i, j, k]).unwrap().into_array().unwrap();
                assert_eq!(check.data, index.data);
            }
        }
    }
}
