use crate::prelude::*;

#[test]
fn max_axis_test_1() {
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
    let module: Module<f32, NoneModel> = ModuleBuilder::new(42).build(NoneModel);
    let tensor =
        Tensor::from_vector_with_shape(vector.clone(), &[4, 2, 3], &module, ReqGrad).unwrap();

    let max = tensor.max_axis(&[0], true, ReqGrad).unwrap();

    max.backward().unwrap();

    tensor
        .grad_vec_eq(&[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        ])
        .unwrap();

    module.reset();
    //
    let tensor =
        Tensor::from_vector_with_shape(vector.clone(), &[4, 2, 3], &module, ReqGrad).unwrap();
    println!("{}", tensor);

    let max = tensor.max_axis(&[0, 2], true, ReqGrad).unwrap();

    max.backward().unwrap();

    tensor
        .grad_vec_eq(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ])
        .unwrap();
}
