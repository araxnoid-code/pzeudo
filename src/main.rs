use pzeudo::*;

struct Model<F> {
    linear_1: Linear<F>,
    linear_2: Linear<F>,
    optim: RMSProp<F>,
}

impl Model<f32> {
    fn forward<ReqGrad>(
        &self,
        requires_grad: ReqGrad,
        dataset: &Tensor<f32, Contiguous, NoGrad>,
        actual: &Tensor<f32, Contiguous, NoGrad>,
    ) -> Result<Tensor<f32, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<f32> + Copy,
    {
        let x = self.linear_1.forward(dataset, requires_grad)?;
        let y = self.linear_2.forward(&x, requires_grad)?;
        let loss = mse(actual, &y, requires_grad)?;
        Ok(loss)
    }
}

fn main() {
    let mut module = Module::<f32>::new(42);
    let mut create_model = module.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 32, WeightInit::Xavier, &mut create_model).unwrap(),
        linear_2: Linear::new(32, 1, WeightInit::Xavier, &mut create_model).unwrap(),
        optim: RMSProp::new(0.01, create_model).unwrap(),
    };

    let shape = [16, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.))
        .collect::<Vec<f32>>();
    let dataset = Tensor::param_from_vector_with_shape(&vector, &shape, &module, NoGrad).unwrap();

    let shape = [16, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = Tensor::param_from_vector_with_shape(&vector, &shape, &module, NoGrad).unwrap();

    let shape = [16, 1];
    let vector = Vec::from_iter(16..shape.iter().product::<usize>() + 16)
        .iter()
        .map(|x| (*x as f32 + 1.))
        .collect::<Vec<f32>>();
    let test = Tensor::param_from_vector_with_shape(&vector, &shape, &module, NoGrad).unwrap();

    let shape = [16, 1];
    let vector = Vec::from_iter(16..shape.iter().product::<usize>() + 16)
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual_test =
        Tensor::param_from_vector_with_shape(&vector, &shape, &module, NoGrad).unwrap();

    let epoch = EpochBuilder::new(100, model, (dataset, test, actual, actual_test));

    module
        .epoch(
            epoch,
            |epoch, _module, model, (dataset, test, actual, actual_test)| {
                println!("epoch: {}", epoch);
                let loss = model.forward(Grad, dataset, actual)?;
                println!("train loss: {}", loss);
                loss.backward()?;

                model.optim.optim()?;
                model.optim.zero_grad();

                let loss = model.forward(NoGrad, test, actual_test)?;
                println!("test loss: {}\n", loss);
                Ok(())
            },
        )
        .unwrap();
}
