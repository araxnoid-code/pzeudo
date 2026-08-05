use crate::prelude::*;

#[test]
fn logic_test_1() {
    let module = Module::<f32>::new(42);

    let shape = [8, 1];
    let vec = (0..shape.iter().product::<usize>())
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let tensor_a = Tensor::param_from_vector_with_shape(&vec, &shape, &module, Grad).unwrap();

    let tensor_b = Tensor::param_from_vector_with_shape(&vec, &shape, &module, Grad).unwrap();

    let tensor_c = tensor_a.mul(&tensor_b, Grad).unwrap();

    let tensor_d = Tensor::param_from_vector_with_shape(&vec, &shape, &module, Grad).unwrap();

    let tensor_e = tensor_d.mul(&tensor_c, Grad).unwrap();
    let tensor_c_no_grad = tensor_c.no_grad().unwrap();

    let tensor_c_grad = tensor_c_no_grad.with_grad().unwrap();

    let tensor_f = tensor_c_grad.mul(&tensor_e, Grad).unwrap();
    tensor_f.backward().unwrap();
}
