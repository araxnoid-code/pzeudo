use crate::prelude::*;

#[test]
fn min_testing_1() {
    let vector = vec![
        0.084775925,
        0.7947035,
        0.11211485,
        0.07567024,
        0.8267454,
        0.6774916,
        0.22457343,
        0.2671346,
        0.08479899,
        0.3399632,
        0.3677261,
        0.4276592,
        0.425839,
        0.9741395,
        0.18334627,
        0.2582574,
        0.64478654,
        0.79133976,
        0.6894156,
        0.4586349,
        0.9188759,
        0.99649936,
        0.53377134,
        0.32139724,
    ];
    let array = Array::<f32>::from_vector_with_shape(vector, &[4, 2, 3]).unwrap();

    let min = array.min().unwrap();
    min.vec_equal(&vec![0.07567024]).unwrap();

    let min_axis_0 = array.min_axis(&[0], true).unwrap();
    min_axis_0
        .vec_equal(&vec![
            0.084775925,
            0.2671346,
            0.08479899,
            0.07567024,
            0.3677261,
            0.32139724,
        ])
        .unwrap();

    let min_axis_1 = array.min_axis(&[1], true).unwrap();
    min_axis_1
        .vec_equal(&vec![
            0.07567024, 0.7947035, 0.11211485, 0.22457343, 0.2671346, 0.08479899, 0.2582574,
            0.64478654, 0.18334627, 0.6894156, 0.4586349, 0.32139724,
        ])
        .unwrap();

    let min_axis_2 = array.min_axis(&[2], true).unwrap();
    min_axis_2
        .vec_equal(&vec![
            0.084775925,
            0.07567024,
            0.08479899,
            0.3399632,
            0.18334627,
            0.2582574,
            0.4586349,
            0.32139724,
        ])
        .unwrap();

    let min_axis_0_1 = array.min_axis(&[0, 1], true).unwrap();
    min_axis_0_1
        .vec_equal(&vec![0.07567024, 0.2671346, 0.08479899])
        .unwrap();

    let min_axis_0_2 = array.min_axis(&[0, 2], true).unwrap();

    min_axis_0_2
        .vec_equal(&vec![0.084775925, 0.07567024])
        .unwrap();

    let min_axis_1_2 = array.min_axis(&[1, 2], true).unwrap();

    min_axis_1_2
        .vec_equal(&vec![0.07567024, 0.08479899, 0.18334627, 0.32139724])
        .unwrap();
}

#[test]
fn argmin_testing_1() {
    let vector = vec![
        0.084775925,
        0.7947035,
        0.11211485,
        0.07567024,
        0.8267454,
        0.6774916,
        0.22457343,
        0.2671346,
        0.08479899,
        0.3399632,
        0.3677261,
        0.4276592,
        0.425839,
        0.9741395,
        0.18334627,
        0.2582574,
        0.64478654,
        0.79133976,
        0.6894156,
        0.4586349,
        0.9188759,
        0.99649936,
        0.53377134,
        0.32139724,
    ];
    let array = Array::<f32>::from_vector_with_shape(vector, &[4, 2, 3]).unwrap();
    let argmin = array.argmin().unwrap();
    argmin.vec_equal(&vec![3.]).unwrap();

    let argmin_axis_0 = array.argmin_axis(&[0], true).unwrap();
    argmin_axis_0
        .vec_equal(&vec![0.0, 1.0, 1.0, 0.0, 1.0, 3.0])
        .unwrap();

    let argmin_axis_1 = array.argmin_axis(&[1], true).unwrap();
    argmin_axis_1
        .vec_equal(&vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ])
        .unwrap();

    let argmin_axis_2 = array.argmin_axis(&[2], true).unwrap();
    argmin_axis_2
        .vec_equal(&vec![0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0])
        .unwrap();

    let argmin_axis_0_1 = array.argmin_axis(&[0, 1], true).unwrap();
    argmin_axis_0_1.vec_equal(&vec![1.0, 2.0, 2.0]).unwrap();

    let argmin_axis_0_2 = array.argmin_axis(&[0, 2], true).unwrap();
    argmin_axis_0_2.vec_equal(&vec![0.0, 0.0]).unwrap();

    let argmin_axis_1_2 = array.argmin_axis(&[1, 2], true).unwrap();
    argmin_axis_1_2
        .vec_equal(&vec![3.0, 2.0, 2.0, 5.0])
        .unwrap();
}
