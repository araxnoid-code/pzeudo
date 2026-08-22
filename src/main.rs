use pzeudo::*;

fn main() {
    let module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);

    let shape = [3, 2, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 0 as f32)
        .collect::<Vec<f32>>();
    let tensor_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 3, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 1 as f32)
        .collect::<Vec<f32>>();
    let tensor_b = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 4, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 2 as f32)
        .collect::<Vec<f32>>();
    let tensor_c = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let shape = [3, 1, 2];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|_| 3 as f32)
        .collect::<Vec<f32>>();
    let tensor_d = Tensor::from_vector_with_shape(&vec_a, &shape, &module_builder, Grad).unwrap();

    let concat = vec![&tensor_a, &tensor_b, &tensor_c, &tensor_d]
        .tensor_concat(1, Grad)
        .unwrap();

    println!("{}", concat);

    let shape = concat.get_shape().to_vec();

    let vec = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let other_tensor = Tensor::from_vector_with_shape(&vec, &shape, &module_builder, Grad).unwrap();
    println!("{}", other_tensor);

    // tensor_b.no_grad().unwrap();
    let mul = other_tensor.mul(&concat, Grad).unwrap();

    mul.backward().unwrap();

    println!("{}", tensor_a.grad_to_string().unwrap());
    println!("=====");
    println!("{}", tensor_c.grad_to_string().unwrap());
    println!("=====");
    println!("{}", tensor_d.grad_to_string().unwrap());
    println!("=====");
}
