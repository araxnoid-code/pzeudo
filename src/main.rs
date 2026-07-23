use pzeudo::{EpochBuilder, Linear, Module, Sgd, mse};
use std::println;

struct Model<F> {
    linear_1: Linear<F>,
    linear_2: Linear<F>,
    optim: Sgd<F>,
}

fn main() {
    let module = Module::<f32>::new();
    let model = Model {
        linear_1: module.new_linear(1, 4).unwrap(),
        linear_2: module.new_linear(4, 1).unwrap(),
        optim: Sgd::new(0.0001, &module),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 10.))
        .collect::<Vec<f32>>();
    let actual = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let epoch = EpochBuilder::new(25, model, (dataset, actual));

    module
        .epoch(epoch, |epoch, module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset).unwrap();
            let y = model.linear_2.forward(&x).unwrap();
            let loss = mse(actual, &y).unwrap();
            println!("loss: {}\n", loss);
            loss.backward().unwrap();

            model.optim.optim().unwrap();
            module.zero_grad();
        })
        .unwrap();
}
