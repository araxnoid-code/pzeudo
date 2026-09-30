use pzeudo::{BatchNorm, ModuleBuilder};

fn main() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let batch_norm = BatchNorm::new(2, 16, &mut model_builder).unwrap();
}
