use pzeudo::{
    Adam, Contiguous, EpochBuilder, Grad, Linear, Module, NoGrad, PzeudoErr, ReqGradTrait, Sgd,
    Tensor, mae, mse,
};

// Create a model
struct Model {
    linear_1: Linear<f32>,
    linear_2: Linear<f32>,
    linear_3: Linear<f32>,
    optim: Sgd<f32>,
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
        let z = self.linear_3.forward(&y, requires_grad)?;
        let loss = mse(target, &z, requires_grad)?;
        Ok(loss)
    }
}

fn main() {
    // Create a module.
    let mut module: Module<f32> = Module::new(42);

    // To initialize a model that has been created, you need a ModelBuilder.
    let mut model_builder = module.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 512, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
        linear_2: Linear::new(512, 256, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
        linear_3: Linear::new(256, 1, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
        optim: Sgd::new(0.01, model_builder).unwrap(),
    };

    // Create training and testing datasets.
    // The dataset below is for testing purposes only.
    let shape = [2046, 1];
    let vec = (0..shape.iter().product::<usize>())
        .map(|i| i as f32)
        .collect::<Vec<f32>>();
    let train_dataset =
        Tensor::param_from_vector_with_shape(&vec, &shape, &module, NoGrad).unwrap();

    let shape = [2046, 1];
    let vec = (0..shape.iter().product::<usize>())
        .map(|i| i as f32 + 10.)
        .collect::<Vec<f32>>();
    let train_target = Tensor::param_from_vector_with_shape(&vec, &shape, &module, NoGrad).unwrap();

    // Initialize EpochBuilder to manage training/testing iterations.
    let epoch = 30;
    let epoch_builder = EpochBuilder::new(
        epoch,
        model,
        (&train_dataset, &train_target, &train_dataset, &train_target),
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

                Ok(())
            },
        )
        .unwrap();
}
