use pzeudo::{BatchNorm, ModuleBuilder, NoneModel, ReqGrad, Tensor};

fn main() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let mut batch_norm = BatchNorm::new(1, 3, &mut model_builder).unwrap();

    let module = module_builder.build(NoneModel);

    let shape = [3, 3, 4];
    let vec = (0..shape.iter().product::<usize>())
        .map(|i| i as f32)
        .collect::<Vec<f32>>();
    let tensor = Tensor::from_vector_with_shape(vec, &shape, &module, ReqGrad).unwrap();

    let norm = batch_norm.forward(&tensor, ReqGrad).unwrap();

    norm.backward().unwrap();
}
