use crate::prelude::*;

#[test]
fn unrecord_test_1() {
    // // Create a module.
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let tensor_a = Tensor::from_vector_with_shape(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 11., 12., 13.],
        &[4, 3],
        &module_builder,
        ReqGrad,
    )
    .unwrap();

    let tensor_b = Tensor::from_vector_with_shape(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 11., 12., 13.],
        &[4, 3],
        &module_builder,
        ReqGrad,
    )
    .unwrap();

    let tensor_c = Tensor::from_vector_with_shape(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 11., 12., 13.],
        &[4, 3],
        &module_builder,
        ReqGrad,
    )
    .unwrap();

    let mut tensor_d = tensor_a.add(&tensor_b, ReqGrad).unwrap();

    let tensor_e = tensor_c.sub(&tensor_d, ReqGrad).unwrap();

    tensor_d.unrecord().unwrap();
    tensor_e.backward().unwrap();

    tensor_e
        .grad_vec_eq(&[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        .unwrap();

    tensor_d
        .grad_vec_eq(&[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
        .unwrap();

    tensor_c
        .grad_vec_eq(&[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        .unwrap();

    tensor_b
        .grad_vec_eq(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        .unwrap();

    tensor_a
        .grad_vec_eq(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        .unwrap();
}
// A------\
//         \
// B--------D <- UnRecord
//           \
// C----------E
