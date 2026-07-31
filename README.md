# Pzeudo
a deep learning project.

## Version 0.0.1
see what's new: [version.md](https://github.com/araxnoid-code/pzeudo/blob/0.0.1/version.md)

## Installation
add the following line to your `Cargo.toml`.
```toml
[dependencies]
pzeudo = "0.0.1"
```
or Run the following Cargo command in your project directory
```sh
cargo add pzeudo
```

## Code
```rust
use pzeudo::{EpochBuilder, Linear, Module, Sgd, mse};

// Create Model
struct Model {
    linear_1: Linear<f32>,
    linear_2: Linear<f32>,
    optim: Sgd<f32>,
}

fn main() {
    // Module Initialization
    // Module Useful in resource management especially Array
    let module = Module::<f32>::new();

    // Initialize the Model That Has Been Created
    let model = Model {
        linear_1: Linear::new(1, 16, &module).unwrap(),
        linear_2: Linear::new(16, 1, &module).unwrap(),
        optim: Sgd::new(0.01, &module),
    };

    // Create Dataset (this is just for example)
    let shape = [8, 1];
    let vec = (0..shape.iter().product::<usize>())
        .map(|x| x as f32 * 0.1)
        .collect::<Vec<f32>>();
    // Function to store tensors permanently
    // (prevents deletion by the arena allocator during the training epoch later)
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vec, &shape)
        .unwrap();

    let shape = [8, 1];
    let vec = (0..shape.iter().product::<usize>())
        .map(|x| x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = module
        .permanent_tensor_from_vector_with_shape(&vec, &shape)
        .unwrap();

    // Create EpochBuilder
    // Functions for operations management in training
    let epoch = 10;
    let arg = (dataset, actual);
    let epoch_builder = EpochBuilder::new(epoch, model, arg);

    module
        .epoch(epoch_builder, |epoch, _module, model, (dataset, actual)| {
            // Training
            let x = model.linear_1.forward(dataset)?;
            let y = model.linear_2.forward(&x)?;
            let loss = mse(actual, &y)?;
            println!("epoch: {epoch}\nloss: {loss}\n");

            // Backpropogation
            loss.backward()?;

            // Optimizer and zero grad
            model.optim.optim()?;
            model.optim.zero_grad();

            Ok(())
        })
        .unwrap();
}
```

to see the development progress: [0.0.1-dev.7_plan.md](https://github.com/araxnoid-code/pzeudo/blob/0.0.1/0.0.1-dev.7_plan.md)
