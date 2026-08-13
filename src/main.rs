use pzeudo::{
    Adam, Contiguous, EpochBuilder, Grad, Linear, Module, ModuleBuilder, NoGrad, OptimizerTrait,
    PzeudoErr, RecordLabel, ReqGradTrait, Tensor, mse,
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
        ReqGrad: ReqGradTrait<f32>,
    {
        let x = self.linear_1.forward(input, requires_grad)?;

        let y = self.linear_2.forward(&x, requires_grad)?;
        let loss = mse(target, &y, requires_grad)?;
        Ok(loss)
    }
}

enum A<F> {
    B(RecordLabel<F>),
    C(RecordLabel<F>),
}

fn main() {

    // // Create a module.
    // let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    // // To initialize a model that has been created, you need a ModelBuilder.
    // let mut model_builder = module_builder.model_builder();
    // let model = Model {
    //     linear_1: Linear::new(1, 16, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
    //     linear_2: Linear::new(16, 1, pzeudo::WeightInit::He, &mut model_builder).unwrap(),
    //     optim: Adam::new(0.01, model_builder).unwrap(),
    // };

    // // Create training and testing datasets.
    // // The dataset below is for testing purposes only.
    // let train_dataset = Tensor::param_from_vector_with_shape(
    //     &[1., 2., 3., 4., 5., 6., 7., 8.],
    //     &[8, 1],
    //     &module_builder,
    //     NoGrad,
    // )
    // .unwrap();
    // let train_target = Tensor::param_from_vector_with_shape(
    //     &[11., 12., 13., 14., 15., 16., 17., 18.],
    //     &[8, 1],
    //     &module_builder,
    //     NoGrad,
    // )
    // .unwrap();

    // let test_dataset = Tensor::param_from_vector_with_shape(
    //     &[9., 10., 11., 12., 13., 14., 15., 16.],
    //     &[8, 1],
    //     &module_builder,
    //     NoGrad,
    // )
    // .unwrap();
    // let test_target = Tensor::param_from_vector_with_shape(
    //     &[19., 20., 21., 22., 23., 24., 25., 26.],
    //     &[8, 1],
    //     &module_builder,
    //     NoGrad,
    // )
    // .unwrap();

    // // Initialize EpochBuilder to manage training/testing iterations.
    // let epoch = 30;
    // let epoch_builder = EpochBuilder::new(
    //     epoch,
    //     (train_dataset, train_target, test_dataset, test_target),
    // );

    // // Use the Module::epoch method to start the iteration.
    // let mut module = module_builder.build(model);
    // module
    //     .epoch(
    //         epoch_builder,
    //         |epoch, _module, model, (train_dataset, train_target, test_dataset, test_target)| {
    //             // training
    //             // Use Grad.
    //             let loss = model.forward(train_dataset, train_target, Grad).unwrap();
    //             println!("epoch:{}\ntrain_loss:{}", epoch, loss);
    //             loss.backward()?;

    //             model.optim.optim()?;
    //             model.optim.zero_grad();

    //             // testing
    //             // Use NoGrad.
    //             let loss = model.forward(test_dataset, test_target, NoGrad).unwrap();
    //             println!("test_loss:{}\n", loss);

    //             Ok(())
    //         },
    //     )
    //     .unwrap();
}
