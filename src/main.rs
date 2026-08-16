use pzeudo::*;

fn main() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let tensor_a = Tensor::from_vector_with_shape(
        &[1., 2., 3., 4., 5., 6.],
        &[2, 1, 3],
        &module_builder,
        Grad,
    )
    .unwrap();

    println!("{}", tensor_a);

    let broadcast = tensor_a.broadcast(&[2, 2, 3]).unwrap();

    println!("{}", broadcast);
}
