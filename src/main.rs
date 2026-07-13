use pzeudo::{
    Array, ArrayTrait, OpsAdd, OpsMatmul2DF64, OpsMatmulNDF32, OpsSlicing, able_broadcast,
    get_broadcast_dim, r, shape_to_stride,
};

fn main() {
    let shape = [2, 1, 3, 1];

    // let array_a = Array::from_vector_with_shape(
    //     &(0..shape.iter().product::<usize>())
    //         .map(|idx| idx as f32)
    //         .collect::<Vec<f32>>(),
    //     &shape,
    // )
    // .unwrap();

    // println!("{}", array_a.to_string());

    // let index = array_a.slicing(&[r(..), r(..2), r(3..3)]).unwrap();
    // println!("{}", index.to_string().unwrap());
}
