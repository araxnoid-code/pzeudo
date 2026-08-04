use crate::prelude::*;

struct Model<F> {
    linear_1: Linear<F>,
    linear_2: Linear<F>,
    optim: Sgd<F>,
}

#[test]
fn linear_model_test_mse_f32() {
    let module = Module::<f32>::new();
    let model = Model {
        linear_1: Linear::new(1, 4, &module).unwrap(),
        linear_2: Linear::new(4, 1, &module).unwrap(),
        optim: Sgd::new(0.0001, &module),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let epoch = EpochBuilder::new(25, model, (dataset, actual));

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset)?;
            let y = model.linear_2.forward(&x)?;
            let loss = mse(actual, &y, Grad)?;
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
    let module = Module::<f64>::new();
    let model = Model {
        linear_1: Linear::new(1, 4, &module).unwrap(),
        linear_2: Linear::new(4, 1, &module).unwrap(),
        optim: Sgd::new(0.0001, &module),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f64 + 1.) * 0.001)
        .collect::<Vec<f64>>();
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f64 + 10.)
        .collect::<Vec<f64>>();
    let actual = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let epoch = EpochBuilder::new(25, model, (dataset, actual));

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset)?;
            let y = model.linear_2.forward(&x)?;
            let loss = mse(actual, &y, Grad)?;
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
    let module = Module::<f32>::new();
    let model = Model {
        linear_1: Linear::new(1, 4, &module).unwrap(),
        linear_2: Linear::new(4, 1, &module).unwrap(),
        optim: Sgd::new(0.0001, &module),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f32 + 1.) * 0.001)
        .collect::<Vec<f32>>();
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f32 + 10.)
        .collect::<Vec<f32>>();
    let actual = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let epoch = EpochBuilder::new(25, model, (dataset, actual));

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset)?;
            let y = model.linear_2.forward(&x)?;
            let loss = mae(actual, &y, Grad)?;
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
    let module = Module::<f64>::new();
    let model = Model {
        linear_1: Linear::new(1, 4, &module).unwrap(),
        linear_2: Linear::new(4, 1, &module).unwrap(),
        optim: Sgd::new(0.0001, &module),
    };

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| (*x as f64 + 1.) * 0.001)
        .collect::<Vec<f64>>();
    let dataset = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let shape = [4, 1];
    let vector = Vec::from_iter(0..shape.iter().product::<usize>())
        .iter()
        .map(|x| *x as f64 + 10.)
        .collect::<Vec<f64>>();
    let actual = module
        .permanent_tensor_from_vector_with_shape(&vector, &shape)
        .unwrap();

    let epoch = EpochBuilder::new(25, model, (dataset, actual));

    module
        .epoch(epoch, |epoch, _module, model, (dataset, actual)| {
            println!("epoch: {}", epoch);
            let x = model.linear_1.forward(dataset)?;
            let y = model.linear_2.forward(&x)?;
            let loss = mae(actual, &y, Grad)?;
            println!("loss: {}\n", loss);
            loss.backward()?;

            model.optim.optim()?;
            model.optim.zero_grad();

            Ok(())
        })
        .unwrap();
}
