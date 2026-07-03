use std::ops::{Add, Neg};

use ndarray::{ArrayD, ArrayViewD, array};
use num_traits::Float;

fn main() {
    let array = array![1.].into_dyn();

    neg(array.view());
}

pub fn neg<F>(arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Neg<Output = F> + Clone,
{
    arr.mapv(|x| -x)
}
