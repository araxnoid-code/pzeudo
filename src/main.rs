use std::ops::{Add, Neg};

use ndarray::{ArrayD, ArrayViewD, array};
use num_traits::Float;
use pzeudo::{PzeudoOpsAdd, Tensor};

fn main() {
    let array = array![1.].into_dyn();

    let tensor = Tensor::from_array(array);
    // tensor.add(rhs, record);
}

pub fn neg<F>(arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Neg<Output = F> + Clone,
{
    arr.mapv(|x| -x)
}
