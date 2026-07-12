use pzeudo::{Array, ArrayTrait, OpsAdd, OpsMatmul2DF64, OpsMatmulNDF32, shape_to_stride};

fn main() {
    let shape = [2, 3, 4, 5, 2];
    let array_a = Array::from_vector_with_shape(
        &(0..shape.iter().product::<usize>())
            .map(|idx| idx as f32)
            .collect::<Vec<f32>>(),
        &shape,
    )
    .unwrap();

    let shape = [2, 3, 5, 4];
    let array_b = Array::from_vector_with_shape(
        &(0..shape.iter().product::<usize>())
            .map(|idx| idx as f32)
            .collect::<Vec<f32>>(),
        &shape,
    )
    .unwrap();

    let array_c = array_a.matmul_nd(&array_b).unwrap();
    println!("{}", array_c.to_string());
}
