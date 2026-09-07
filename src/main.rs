use pzeudo::{
    Adam, EpochBuilder, Linear, ModuleBuilder, OptimizerTrait, ReqGrad, ReqNoGrad, Tensor,
    cross_entropy_loss, mse, relu, sigmoid, softmax, softmax_cross_entropy,
};

struct Model {
    linear_a: Linear<f32>,
    linear_b: Linear<f32>,
    linear_c: Linear<f32>,
    optim: Adam<f32>,
}

fn main() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let model = Model {
        linear_a: Linear::new(1, 16, pzeudo::WeightInit::Xavier, &mut model_builder).unwrap(),
        linear_b: Linear::new(16, 32, pzeudo::WeightInit::Xavier, &mut model_builder).unwrap(),
        linear_c: Linear::new(32, 2, pzeudo::WeightInit::Xavier, &mut model_builder).unwrap(),
        optim: Adam::new(0.1, model_builder).unwrap(),
    };

    let vec = vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    ];
    let train =
        Tensor::param_from_vector_with_shape(vec, &[20, 1], &module_builder, ReqNoGrad).unwrap();

    let vec = vec![
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
        1., 0., //
        0., 1., //
    ];
    let target =
        Tensor::param_from_vector_with_shape(vec, &[20, 2], &module_builder, ReqNoGrad).unwrap();

    let mut module = module_builder.build(model);
    let epoch_builder = EpochBuilder::new(20, (train, target));

    module
        .epoch(epoch_builder, |epoch, _, model, (train, target)| {
            println!("epoch: {}", epoch);
            let a = model.linear_a.forward(train, ReqGrad)?;
            let b = relu(&a, ReqGrad)?;
            let c = model.linear_b.forward(&b, ReqGrad)?;
            let d = relu(&c, ReqGrad)?;
            let e = model.linear_c.forward(&d, ReqGrad)?;

            // let prop = softmax(&e, 1, ReqGrad)?;

            // let loss = cross_entropy_loss(target, &prop, ReqGrad)?;
            let loss = softmax_cross_entropy(target, &e, 1, ReqGrad)?;
            println!("loss: {}", loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();
            // println!();
            Ok(())
        })
        .unwrap();
}
