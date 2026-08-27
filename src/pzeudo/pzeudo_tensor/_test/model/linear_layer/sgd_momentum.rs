use crate::prelude::*;

struct Model<F> {
    linear_1: Linear<F>,
    linear_2: Linear<F>,
    optim: SgdMomentum<F>,
}

#[test]
fn linear_model_test_mse_f32() {
    let mut module = ModuleBuilder::<f32>::new(42);

    let mut create_model = module.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 4, WeightInit::He, &mut create_model).unwrap(),
        linear_2: Linear::new(4, 1, WeightInit::He, &mut create_model).unwrap(),
        optim: SgdMomentum::new(0.0001, create_model).unwrap(),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let dataset = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqGrad).unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let epoch = EpochBuilder::new(25, (dataset, actual));
    let mut module = module.build(model);

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset, ReqGrad)?;
            let y = model.linear_2.forward(&x, ReqGrad)?;
            let loss = mse(actual, &y, ReqGrad)?;
            println!("loss: {}\n", loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();
            Ok(())
        })
        .unwrap();
}

#[test]
fn linear_model_test_mse_f64() {
    let mut module = ModuleBuilder::<f64>::new(42);
    let mut create_model = module.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 4, WeightInit::He, &mut create_model).unwrap(),
        linear_2: Linear::new(4, 1, WeightInit::He, &mut create_model).unwrap(),
        optim: SgdMomentum::new(0.0001, create_model).unwrap(),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f64 + 1.) * 0.001)
        .collect::<Vec<f64>>();
    let dataset =
        Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f64 + 10.)
        .collect::<Vec<f64>>();
    let actual = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqGrad).unwrap();

    let epoch = EpochBuilder::new(25, (dataset, actual));
    let mut module = module.build(model);

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset, ReqGrad)?;
            let y = model.linear_2.forward(&x, ReqGrad)?;
            let loss = mse(actual, &y, ReqGrad)?;
            println!("loss: {}\n", loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();
            Ok(())
        })
        .unwrap();
}

#[test]
fn linear_model_test_mae_f32() {
    let mut module = ModuleBuilder::<f32>::new(42);
    let mut create_model = module.model_builder();

    let model = Model {
        linear_1: Linear::new(1, 4, WeightInit::He, &mut create_model).unwrap(),
        linear_2: Linear::new(4, 1, WeightInit::He, &mut create_model).unwrap(),
        optim: SgdMomentum::new(0.0001, create_model).unwrap(),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let dataset = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqGrad).unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqGrad).unwrap();

    let epoch = EpochBuilder::new(25, (dataset, actual));
    let mut module = module.build(model);

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset, ReqGrad)?;
            let y = model.linear_2.forward(&x, ReqGrad)?;
            let loss = mae(actual, &y, ReqGrad)?;
            println!("loss: {}\n", loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();
            Ok(())
        })
        .unwrap();
}

#[test]
fn linear_model_test_mae_f64() {
    let mut module = ModuleBuilder::<f64>::new(42);
    let mut create_model = module.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 4, WeightInit::Xavier, &mut create_model).unwrap(),
        linear_2: Linear::new(4, 1, WeightInit::Xavier, &mut create_model).unwrap(),
        optim: SgdMomentum::new(0.0001, create_model).unwrap(),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f64 + 1.) * 0.001)
        .collect::<Vec<f64>>();
    let dataset =
        Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f64 + 10.)
        .collect::<Vec<f64>>();
    let actual = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let epoch = EpochBuilder::new(25, (dataset, actual));
    let mut module = module.build(model);

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset, ReqGrad)?;
            let y = model.linear_2.forward(&x, ReqGrad)?;
            let loss = mae(actual, &y, ReqGrad)?;
            println!("loss: {}\n", loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();

            Ok(())
        })
        .unwrap();
}

impl Model<f32> {
    fn forward<ReqGrad>(
        &self,
        requires_grad: ReqGrad,
        dataset: &Tensor<f32, Contiguous, ReqNoGrad>,
        actual: &Tensor<f32, Contiguous, ReqNoGrad>,
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

#[test]
fn linear_model_test_mse_f32_train_eval() {
    let mut module = ModuleBuilder::<f32>::new(42);
    let mut create_model = module.model_builder();

    let model = Model {
        linear_1: Linear::new(1, 16, WeightInit::He, &mut create_model).unwrap(),
        linear_2: Linear::new(16, 1, WeightInit::He, &mut create_model).unwrap(),
        optim: SgdMomentum::new(0.01, create_model).unwrap(),
    };

    let shape = [16, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let dataset =
        Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let shape = [16, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let shape = [16, 1];
    let vector = Vec::from_iter(16..shape.iter().product::<usize>() + 16)
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let test = Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let shape = [16, 1];
    let vector = Vec::from_iter(16..shape.iter().product::<usize>() + 16)
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual_test =
        Tensor::param_from_vector_with_shape(&vector, &shape, &module, ReqNoGrad).unwrap();

    let epoch = EpochBuilder::new(50, (dataset, test, actual, actual_test));
    let mut module = module.build(model);

    module
        .epoch(
            epoch,
            |epoch, _module, model, (dataset, test, actual, actual_test)| {
                println!("epoch: {}", epoch);
                let loss = model.forward(ReqGrad, dataset, actual)?;
                println!("train loss: {}", loss);
                loss.backward()?;

                model.optim.optim()?;
                model.optim.zero_grad();

                let loss = model.forward(ReqNoGrad, test, actual_test)?;
                println!("test loss: {}\n", loss);
                Ok(())
            },
        )
        .unwrap();
}
