use pzeudo::{EpochBuilder, Grad, Linear, Module, NoGrad, Sgd, mse};

fn main() {
    let module = Module::<f32>::new();

    let shape = [8, 1];
    let vec = (0..shape.iter().product::<usize>())
        .map(|x| x as f32 * 0.1)
        .collect::<Vec<f32>>();

    let mut dataset = module
        .tensor_from_vector_with_shape::<NoGrad>(&vec, &shape)
        .unwrap();

    dataset.backward().unwrap();

    // dataset.no_grad().unwrap();
}
