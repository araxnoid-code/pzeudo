use crate::prelude::*;

#[test]
fn max_test_1() {
    let vec = vec![
        0.6493464, 0.6015608, 0.21999699, 0.6243996, 0.2509725, 0.53029996, 0.6398015, 0.34393674,
        0.81197757, 0.8566167, 0.20921028, 0.66543806, 0.07820165, 0.33682317, 0.28257972,
        0.5195835, 0.73698926, 0.6886583,
    ];

    let module: Module<f32, NoneModel> = ModuleBuilder::new(42).build(NoneModel);
    let tensor = Tensor::from_vector_with_shape(vec, &[3, 2, 3], &module, ReqGrad).unwrap();

    let max = tensor.max(ReqGrad).unwrap();

    max.backward().unwrap();
    tensor
        .grad_vec_eq(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ])
        .unwrap();
}
