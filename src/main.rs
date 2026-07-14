use pzeudo::{
    Array, ArrayTrait, OpsAdd, OpsBroadcast, OpsMatmul2DF64, OpsMatmulNDF32, OpsPermute,
    OpsSlicing, OpsSum, able_broadcast, get_broadcast_dim, r, shape_to_stride,
};

fn main() {
    let shape = [2, 2, 3];
    let array_a = Array::from_vector_with_shape(
        &(0..shape.iter().product::<usize>())
            .map(|idx| idx as f32)
            .collect::<Vec<f32>>(),
        &shape,
    )
    .unwrap();
    println!("{}", array_a.to_string());

    let data = array_a.sum_axis(&[0, 1, 2], false).unwrap();
    println!("{}", data.to_string());
}
