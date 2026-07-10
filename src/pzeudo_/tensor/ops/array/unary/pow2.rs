use std::ops::Mul;

use ndarray::{ArrayD, ArrayViewD};

pub fn pow2<F>(arr: ArrayViewD<'_, F>) -> ArrayD<F>
where
    F: Clone + Mul<Output = F> + Copy,
{
    arr.mapv(|x| x * x)
}
