use pzeudo::*;

fn main() {
    let module = ModuleBuilder::<f64>::new(42).build(NoneModel);

    let shape = [6];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_a = Tensor::from_vector_with_shape(vec_a, &shape, &module, ReqGrad).unwrap();

    let shape = [6];
    let vec_b = (5..shape.iter().product::<usize>() + 5)
        .map(|idx| idx as f64)
        .collect::<Vec<f64>>();
    let array_b = Tensor::from_vector_with_shape(vec_b, &shape, &module, ReqGrad).unwrap();

    array_a.dot(&array_b, ReqGrad).unwrap();
}
