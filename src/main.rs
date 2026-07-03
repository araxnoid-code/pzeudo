use std::ops::{Add, Div, Neg};

use ndarray::{ArrayD, ArrayViewD, array};
use num_traits::Float;
use pzeudo::{Backward, PzeudoOpsAdd, PzeudoOpsDiv, Tensor, TensorView};

fn main() {
    // let mut record = vec![];
    let array = array![1.].into_dyn();

    let test = Tensor::from_array(array);
}
