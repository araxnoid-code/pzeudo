use pzeudo::*;

struct Model<F> {
    linear_1: Linear<F>,
    linear_2: Linear<F>,
    optim: RMSProp<F>,
}

fn main() {
    let mut module_builder = ModuleBuilder::<f32>::new(42);

    let mut create_model = module_builder.model_builder();
    let model = Model {
        linear_1: Linear::new(1, 4, WeightInit::HeIn, &mut create_model).unwrap(),
        linear_2: Linear::new(4, 1, WeightInit::HeIn, &mut create_model).unwrap(),
        optim: RMSProp::new(0.01, create_model).unwrap(),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.1)
        .collect::<Vec<f32>>();
    let dataset =
        Tensor::param_from_vector_with_shape(vector, &shape, &module_builder, ReqGrad).unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual =
        Tensor::param_from_vector_with_shape(vector, &shape, &module_builder, ReqNoGrad).unwrap();

    let epoch = EpochBuilder::new(100, (dataset, actual));
    let mut module = module_builder.build(model);

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
