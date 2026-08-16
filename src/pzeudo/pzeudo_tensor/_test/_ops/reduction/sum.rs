use crate::prelude::*;

#[test]
fn sum_test_1() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let tensor_a = Tensor::from_vector_with_shape(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 11., 12., 13.],
        &[2, 2, 3],
        &module_builder,
        Grad,
    )
    .unwrap();

    let tensor_b = Tensor::from_vector_with_shape(&[10.], &[1], &module_builder, Grad).unwrap();

    let tensor_sum = tensor_a.sum(Grad).unwrap();

    let tensor_mul = tensor_b.mul(&tensor_sum, Grad).unwrap();
    tensor_mul.backward().unwrap();

    tensor_a
        .grad_vec_eq(&[10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.])
        .unwrap();
}
