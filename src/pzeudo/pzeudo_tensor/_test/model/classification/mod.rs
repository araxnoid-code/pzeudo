use crate::prelude::*;

struct Model {
    linear_a: Linear<f32>,
    linear_b: Linear<f32>,
    optim: Adam<f32>,
}

#[test]
fn classification_test_1() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let model = Model {
        linear_a: Linear::new(1, 8, WeightInit::He, &mut model_builder).unwrap(),
        linear_b: Linear::new(8, 2, WeightInit::He, &mut model_builder).unwrap(),
        optim: Adam::new(1., model_builder).unwrap(),
    };

    let mut vec_test = vec![];
    let train_vec = (0..64)
        .map(|i| {
            if (i % 2) == 0 {
                vec_test.push(0.);
                vec_test.push(1.);
            } else {
                vec_test.push(1.);
                vec_test.push(0.);
            };

            i as f32 / 32.
        })
        .collect::<Vec<f32>>();
    let train =
        Tensor::param_from_vector_with_shape(&train_vec, &[64, 1], &module_builder, ReqNoGrad)
            .unwrap();

    let target =
        Tensor::param_from_vector_with_shape(&vec_test, &[64, 2], &module_builder, ReqNoGrad)
            .unwrap();

    let mut module = module_builder.build(model);
    let epoch_builder = EpochBuilder::new(100, (train, target));

    module
        .epoch(epoch_builder, |_epoch, _module, model, (train, target)| {
            let a = model.linear_a.forward(train, ReqGrad)?;
            let b = softplus(&a, ReqGrad)?;
            let c = model.linear_b.forward(&b, ReqGrad)?;

            let prop = softmax(&c, 1, ReqGrad)?;

            let loss = cross_entropy_loss(target, &prop, ReqGrad)?;

            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();

            Ok(())
        })
        .unwrap();
}
