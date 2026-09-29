use crate::prelude::*;

#[test]
fn dot_f32_test_1() {
    let module = ModuleBuilder::<f32>::new(42).build(NoneModel);

    let shape = [6];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Tensor::from_vector_with_shape(vec_a, &shape, &module, ReqGrad).unwrap();

    let shape = [6];
    let vec_b = (5..shape.iter().product::<usize>() + 5)
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Tensor::from_vector_with_shape(vec_b, &shape, &module, ReqGrad).unwrap();

    let dot = array_a.dot(&array_b, ReqGrad).unwrap();

    dot.backward().unwrap();

    array_a
        .grad_vec_eq(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        .unwrap();
    array_b
        .grad_vec_eq(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
}
