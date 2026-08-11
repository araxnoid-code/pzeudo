# Pzeudo
a deep learning project.

## Version 0.0.2
see what's new: [version.md](https://github.com/araxnoid-code/pzeudo/blob/0.0.2/version.md)

## Code
```rust
use pzeudo::{
    Adam, Contiguous, EpochBuilder, Grad, Linear, Module, NoGrad, PzeudoErr, ReqGradTrait, Tensor,
    mse,
};

// Create a model
struct Model {
    linear_1: Linear<f32>,
    linear_2: Linear<f32>,
    optim: Adam<f32>,
}

// forward pass in the model
impl Model {
    fn forward<ReqGrad>(
        &self,
        input: &Tensor<f32, Contiguous, NoGrad>,
        target: &Tensor<f32, Contiguous, NoGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<f32, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: Copy + Clone + ReqGradTrait<f32>,
    {
        let x = self.linear_1.forward(input, requires_grad)?;
        let y = self.linear_2.forward(&x, requires_grad)?;
        let loss = mse(target, &y, requires_grad)?;
        Ok(loss)
    }
}

fn main() {
    // Create a module.
    let mut module: Module<f32> = Module::new(42);

    // To initialize a model that has been created, you need a ModelBuilder.
    let mut model_builder = module.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 16, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
        linear_2: Linear::new(16, 1, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
        optim: Adam::new(0.01, model_builder).unwrap(),
    };

    // Create training and testing datasets.
    // The dataset below is for testing purposes only.
    let train_dataset = Tensor::param_from_vector_with_shape(
        &[1., 2., 3., 4., 5., 6., 7., 8.],
        &[8, 1],
        &module,
        NoGrad,
    )
    .unwrap();
    let train_target = Tensor::param_from_vector_with_shape(
        &[11., 12., 13., 14., 15., 16., 17., 18.],
        &[8, 1],
        &module,
        NoGrad,
    )
    .unwrap();

    let test_dataset = Tensor::param_from_vector_with_shape(
        &[9., 10., 11., 12., 13., 14., 15., 16.],
        &[8, 1],
        &module,
        NoGrad,
    )
    .unwrap();
    let test_target = Tensor::param_from_vector_with_shape(
        &[19., 20., 21., 22., 23., 24., 25., 26.],
        &[8, 1],
        &module,
        NoGrad,
    )
    .unwrap();

    // Initialize EpochBuilder to manage training/testing iterations.
    let epoch = 30;
    let epoch_builder = EpochBuilder::new(
        epoch,
        model,
        (train_dataset, train_target, test_dataset, test_target),
    );

    // Use the Module::epoch method to start the iteration.
    module
        .epoch(
            epoch_builder,
            |epoch, _module, model, (train_dataset, train_target, test_dataset, test_target)| {
                // training
                // Use Grad.
                let loss = model.forward(train_dataset, train_target, Grad).unwrap();
                println!("epoch:{}\ntrain_loss:{}", epoch, loss);
                loss.backward()?;

                model.optim.optim()?;
                model.optim.zero_grad();

                // testing
                // Use NoGrad.
                let loss = model.forward(test_dataset, test_target, NoGrad).unwrap();
                println!("test_loss:{}\n", loss);

                Ok(())
            },
        )
        .unwrap();
}
```

### Save and Load
#### Save
Based on the code above, modify the model initialization section.
```rust
// ...
// make it mutable
let mut model = Model {
    linear_1: Linear::new(1, 16, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
    linear_2: Linear::new(16, 1, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
    optim: Adam::new(0.01, model_builder).unwrap(),
};
// ...
```
Then, in the EpochBuilder section, borrow mut the model instead of moving it.
```rust
// ...
let epoch = 30;
let epoch_builder = EpochBuilder::new(
    epoch,
    &mut model, // borrow mut
    (train_dataset, train_target, test_dataset, test_target),
);
// ...
```
Then, after training is complete, add the line:
```rust
// ...
module
    .epoch(
        epoch_builder,
        |epoch, _module, model, (train_dataset, train_target, test_dataset, test_target)| {
            // training
            // Use Grad.
            let loss = model.forward(train_dataset, train_target, Grad).unwrap();
            println!("epoch:{}\ntrain_loss:{}", epoch, loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();

            // testing
            // Use NoGrad.
            let loss = model.forward(test_dataset, test_target, NoGrad).unwrap();
            println!("test_loss:{}\n", loss);

            Ok(())
        },
    )
    .unwrap();

// save params
model.optim.save_params("params.json").unwrap();
```

#### Load
Ensure the model architecture matches the model whose parameters were saved. Once that is done:
```rust
// ...
let mut model_builder = module.model_builder();
// load params
model_builder.load_params("params.json").unwrap();
let mut model = Model {
    linear_1: Linear::new(1, 16, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
    linear_2: Linear::new(16, 1, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
    optim: Adam::new(0.01, model_builder).unwrap(),
};
// ...
```

to see the development progress: [0.0.2_plan.md](https://github.com/araxnoid-code/pzeudo/blob/0.0.2/0.0.2_plan.md)
