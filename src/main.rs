use pzeudo::{
    Array, ArrayTrait, OpsAdd, OpsAvg, OpsBroadcast, OpsDotProductF32, OpsMatmul2DF64,
    OpsMatmulNDF32, OpsPermute, OpsSlicing, OpsSum, able_broadcast, get_broadcast_dim, r,
    shape_to_stride,
};

fn main() {
    let shape = [10];
    let array_a = Array::from_vector_with_shape(
        &(0..shape.iter().product::<usize>())
            .map(|idx| idx)
            .collect::<Vec<usize>>(),
        &shape,
    )
    .unwrap();
    println!("{}", array_a.to_string());

    let shape = [10];
    let array_b = Array::from_vector_with_shape(
        &(0..shape.iter().product::<usize>())
            .map(|idx| idx as f32)
            .collect::<Vec<f32>>(),
        &shape,
    )
    .unwrap();
    println!("{}", array_b.to_string());

    let sum = array_a.avg().unwrap();
    println!("{}", sum.to_string());
    // let dot = array_a.dot_f32(&array_b).unwrap();
    // println!("{}", dot.to_string());

    // let data = array_a.sum_axis(&[0, 1, 2], false).unwrap();
    // println!("{}", data.to_string());

    // let data = [10., 20.];
    // let len = data.iter().product::<f32>();
    // let data = 10;
    // let data: f32 = data.;
}
