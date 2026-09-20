use crate::prelude::*;
#[test]
fn max_test_1() {
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

    let max = array.max().unwrap();
    max.vec_equal(&vec![0.99649936]).unwrap();

    let max_axis_0 = array.max_axis(&[0], true).unwrap();
    max_axis_0
        .vec_equal(&vec![
            0.6894156, 0.9741395, 0.9188759, 0.99649936, 0.8267454, 0.79133976,
        ])
        .unwrap();

    let max_axis_1 = array.max_axis(&[1], true).unwrap();
    max_axis_1
        .vec_equal(&vec![
            0.084775925,
            0.8267454,
            0.6774916,
            0.3399632,
            0.3677261,
            0.4276592,
            0.425839,
            0.9741395,
            0.79133976,
            0.99649936,
            0.53377134,
            0.9188759,
        ])
        .unwrap();

    let max_axis_2 = array.max_axis(&[2], true).unwrap();
    max_axis_2
        .vec_equal(&vec![
            0.7947035, 0.8267454, 0.2671346, 0.4276592, 0.9741395, 0.79133976, 0.9188759,
            0.99649936,
        ])
        .unwrap();

    let max_axis_0_1 = array.max_axis(&[0, 1], true).unwrap();
    max_axis_0_1
        .vec_equal(&vec![0.99649936, 0.9741395, 0.9188759])
        .unwrap();

    let max_axis_0_2 = array.max_axis(&[0, 2], true).unwrap();
    max_axis_0_2
        .vec_equal(&vec![0.9741395, 0.99649936])
        .unwrap();

    let max_axis_1_2 = array.max_axis(&[1, 2], true).unwrap();
    max_axis_1_2
        .vec_equal(&vec![0.8267454, 0.4276592, 0.9741395, 0.99649936])
        .unwrap();
}
