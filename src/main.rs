use pzeudo::{
    Array, ArrayTrait, OpsAdd, OpsBroadcast, OpsMatmul2DF64, OpsMatmulNDF32, OpsPermute,
    OpsSlicing, able_broadcast, get_broadcast_dim, r, shape_to_stride,
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

    let per = array_a.t();

    println!("{}", per.to_string().unwrap());

    // shape [2, 1, 4]
    // stride [4, 4, 1]
}
