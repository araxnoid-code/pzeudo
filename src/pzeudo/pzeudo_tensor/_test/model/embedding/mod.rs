use crate::prelude::*;

struct Model {
    embedding: Embedding<f32, ReqGrad>,
    linear: Linear<f32>,
    optim: Adam<f32>,
}

#[test]
fn embedding_test_1() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let model = Model {
        embedding: Embedding::new(10, 16, &mut model_builder, ReqGrad).unwrap(),
        linear: Linear::new(16, 1, WeightInit::He, &mut model_builder).unwrap(),
        optim: Adam::new(0.01, model_builder).unwrap(),
    };

    let train: Vec<f32> = vec![0., 2., 5., 2., 3., 3., 0., 7., 7., 6.];
    let train_tensor =
        Tensor::param_from_vector_with_shape(&train, &[train.len()], &module_builder, ReqNoGrad)
            .unwrap();

    let test: Vec<f32> = train.iter().map(|val| *val + 1.).collect();
    let test_tensor =
        Tensor::param_from_vector_with_shape(&test, &[test.len(), 1], &module_builder, ReqNoGrad)
            .unwrap();

    let mut module = module_builder.build(model);
    let epoch_builder = EpochBuilder::new(25, (train_tensor, test_tensor));
    module
        .epoch(epoch_builder, |_, _, model, (train, target)| {
            let x = model.embedding.forward(train, ReqGrad)?;
            let x = model.linear.forward(&x, ReqGrad)?;
            let loss = mse(target, &x, ReqGrad)?;

            loss.backward()?;
            model.optim.optim()?;

            Ok(())
        })
        .unwrap();
}
