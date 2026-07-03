use std::ops::{Add, Neg};

use ndarray::{ArrayD, ArrayViewD, array};
use num_traits::Float;
use pzeudo::{Backward, PzeudoOpsAdd, Tensor, TensorView};

fn main() {
    // let mut record = vec![];
    let array = array![1.].into_dyn();

    let test = Tensor::from_array(array);
}

pub fn neg<F>(arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Neg<Output = F> + Clone,
{
    arr.mapv(|x| -x)
}
