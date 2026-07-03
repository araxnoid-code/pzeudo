use std::ops::Neg;

use ndarray::{ArrayD, ArrayViewD};

pub fn neg<F>(arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Neg<Output = F> + Clone,
{
    arr.mapv(|x| -x)
}
