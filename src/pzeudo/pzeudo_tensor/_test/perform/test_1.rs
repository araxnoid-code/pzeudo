use pzeudo::*;

struct Model {
    linear_a: Linear<f32>,
    linear_b: Linear<f32>,
    norm_a: LayerNorm<f32, ReqGrad>,
    linear_c: Linear<f32>,
    linear_d: Linear<f32>,
    norm_b: LayerNorm<f32, ReqGrad>,
    dropout: Dropout,
    linear_e: Linear<f32>,
    optim: Adam<f32>,
}

impl Model {
    pub fn forward<Phase>(
        &mut self,
        input: &Tensor<f32, Contiguous, ReqNoGrad>,
        target: &Tensor<f32, Contiguous, ReqNoGrad>,
        phase: Phase,
    ) -> Result<(), PzeudoErr>
    where
        Phase: PhaseStatus<f32>,
    {
        let a = relu(&self.linear_a.forward(input, phase)?, phase)?;
        let b = relu(&self.linear_b.forward(&a, phase)?, phase)?;
        let norm = self.norm_a.forward(&b, phase)?;
        let c = relu(&self.linear_c.forward(&norm, phase)?, phase)?;
        let d = relu(&self.linear_d.forward(&c, phase)?, phase)?;
        let norm = self.norm_b.forward(&d.add(&a, phase)?, phase)?;
        let e = self.linear_e.forward(&norm, phase)?;

        let pred = softmax(&e, 1, phase)?;
        let loss = cross_entropy_loss(target, &pred, phase)?;

        loss.backward()?;
        println!("train loss: {}", loss);

        println!("{}", self.linear_e.get_bias().grad_to_string()?);

        Ok(())
    }
}

fn main() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let model = Model {
        linear_a: Linear::new(1, 512, WeightInit::He, &mut model_builder).unwrap(),
        linear_b: Linear::new(512, 1024, WeightInit::He, &mut model_builder).unwrap(),
        norm_a: LayerNorm::new(Some(1024), &mut model_builder, ReqGrad).unwrap(),
        linear_c: Linear::new(1024, 2048, WeightInit::He, &mut model_builder).unwrap(),
        linear_d: Linear::new(2048, 512, WeightInit::He, &mut model_builder).unwrap(),
        norm_b: LayerNorm::new(Some(512), &mut model_builder, ReqGrad).unwrap(),
        dropout: Dropout::new(0.7, 42),
        linear_e: Linear::new(512, 2, WeightInit::He, &mut model_builder).unwrap(),
        optim: Adam::new(0.01, model_builder).unwrap(),
    };

    let mut vec_test = vec![];
    let train_vec = (0..32)
        .map(|i| {
            if (i % 2) == 0 {
                vec_test.push(0.);
                vec_test.push(1.);
            } else {
                vec_test.push(1.);
                vec_test.push(0.);
            };

            i as f32 / 1.
        })
        .collect::<Vec<f32>>();
    let train =
        Tensor::param_from_vector_with_shape(&train_vec, &[32, 1], &module_builder, ReqNoGrad)
            .unwrap();

    let target =
        Tensor::param_from_vector_with_shape(&vec_test, &[32, 2], &module_builder, ReqNoGrad)
            .unwrap();

    let mut module = module_builder.build(model);
    let epoch_builder = EpochBuilder::new(100, (train, target));

    module
        .epoch(epoch_builder, |_epoch, _module, model, (train, target)| {
            println!("{}", _epoch);
            let loss = model.forward(train, target, TrainPhase)?;
            // println!("train loss: {}", loss);
            // loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();

            // let loss = model.forward(train, target, EvalPhase)?;
            // println!("Eval loss: {}", loss);

            println!();
            Ok(())
        })
        .unwrap();
}
