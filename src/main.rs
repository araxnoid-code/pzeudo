use pzeudo::*;

fn main() {
    let module = Module::<f32>::new(42);

    let shape = [3, 2, 1];
    let vec_a = (0..shape.iter().product::<usize>())
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_a = Tensor::from_vector_with_shape(&vec_a, &shape, &module, Grad).unwrap();

    let shape = [3, 2, 3];
    let vec_b = (5..shape.iter().product::<usize>() + 5)
        .map(|idx| idx as f32)
        .collect::<Vec<f32>>();
    let array_b = Tensor::from_vector_with_shape(&vec_b, &shape, &module, Grad).unwrap();

    let div = array_a.div(&array_b, Grad).unwrap();
    div.backward().unwrap();

    // Check
    // let div_value = [0.0, 0.16666667, 0.2857143, 0.375, 0.44444445, 0.5];
    // div.value_vec_eq(&div_value).unwrap();

    // println!("{}", array_a.grad_to_string().unwrap());
    // let a_grad = [0.2, 0.16666667, 0.14285715, 0.125, 0.11111111, 0.1];
    // array_a.grad_vec_eq(&a_grad).unwrap();

    // let b_grad = [
    //     0.0,
    //     -0.027777778,
    //     -0.040816326,
    //     -0.046875,
    //     -0.049382716,
    //     -0.05,
    // ];
    // array_b.grad_vec_eq(&b_grad).unwrap();
}
