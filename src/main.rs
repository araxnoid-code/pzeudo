use std::{cell::RefCell, rc::Rc};

use ndarray::array;
use pzeudo::Tensor;

fn main() {
    let array = array![[1., 2., 3.]].into_dyn();
    let grad = array![[1., 2., 3.]].into_dyn();
    let tensor = Tensor {
        array,
        gradient: Some(Rc::new(RefCell::new(grad))),
    };
}

// use std::ops::AddAssign;

// use ndarray::{ArrayD, ArrayViewD, Axis, array, concatenate, linalg::Dot, s};
// use pzeudo::{Backward, PzeudoOpsAdd, TensorF32, TensorTrait, able_broadcast};

// fn main() {
//     let mut record = vec![];
//     let tensor_a = TensorF32::from_array(ArrayD::<f32>::ones(vec![2, 1, 5]));
//     let tensor_b = TensorF32::from_array(ArrayD::<f32>::ones(vec![5, 2, 3, 5]));

//     let tensor_c = tensor_a.add(&tensor_b, &mut record).unwrap();
//     println!("{}", tensor_c);

//     tensor_c.backward(&mut record);

//     // let grad = tensor_a.get_share_gradient().unwrap();
//     // println!("{}", grad.borrow());

//     // let array_a = ArrayD::<f32>::from_shape_vec(
//     //     vec![3, 2, 3, 4],
//     //     (0..3 * 2 * 3 * 4)
//     //         .into_iter()
//     //         .map(|idx| idx as f32)
//     //         .collect(),
//     // )
//     // .unwrap();
//     // println!("{}\n", array_a);

//     // let array_b = ArrayD::<f32>::from_shape_vec(
//     //     vec![3, 2, 4, 3],
//     //     (3 * 2 * 3 * 4..(3 * 2 * 3 * 4) + (3 * 2 * 4 * 3))
//     //         .into_iter()
//     //         .map(|idx| idx as f32)
//     //         .collect(),
//     // )
//     // .unwrap();
//     // // println!("{}\n", array_b);

//     // // matmul nd (minimal dim adalah 3)
//     // let len_shape = array_a.shape().len();
//     // let mut output_shape = array_a.shape()[..len_shape - 2].to_vec();
//     // output_shape.push(array_a.shape()[len_shape - 2]);
//     // output_shape.push(array_b.shape()[len_shape - 1]);
//     // let batch = array_a.shape()[..len_shape - 2].iter().product::<usize>();

//     // let mut new_shape = vec![batch];
//     // new_shape.extend_from_slice(&array_a.shape()[len_shape - 2..]);
//     // let array_a_reshape = array_a.to_shape(new_shape).unwrap();

//     // let mut new_shape = vec![batch];
//     // new_shape.extend_from_slice(&array_b.shape()[len_shape - 2..]);
//     // let array_b_reshape = array_b.to_shape(new_shape).unwrap();

//     // let mut result = ArrayD::<f32>::zeros(vec![
//     //     batch,
//     //     array_a.shape()[len_shape - 2],
//     //     array_b.shape()[len_shape - 1],
//     // ]);

//     // array_a_reshape
//     //     .axis_iter(Axis(0))
//     //     .zip(array_b_reshape.axis_iter(Axis(0)))
//     //     .enumerate()
//     //     .for_each(|(idx, (array_a, array_b))| {
//     //         let mut slice = result.slice_mut(s![idx, .., ..]);
//     //         slice.assign(&array_a.dot(&array_b));
//     //     });

//     // let array = result.to_shape(output_shape).unwrap();
//     // println!("{}", array);
// }
