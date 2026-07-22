use pzeudo::{Linear, Module};
use std::println;

struct Model<F> {
    linear_1: Linear<F>,
    linear_2: Linear<F>,
}

fn main() {
    let module = Module::<f32>::new();
    let model = Model {
        linear_1: module.new_linear(8, 4).unwrap(),
        linear_2: module.new_linear(4, 2).unwrap(),
    };

    let shape = [16, 8];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    module
        .epoch(10, model, dataset, |epoch, _module, model, dataset| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(&dataset).unwrap();
            let y = model.linear_2.forward(&x).unwrap();
            y.backward().unwrap();
        })
        .unwrap();
}
