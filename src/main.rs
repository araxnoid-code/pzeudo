use std::ops::{Add, Neg};

use ndarray::{ArrayD, ArrayViewD, array};
use num_traits::Float;
use pzeudo::{PzeudoOpsAdd, Tensor, TensorView};

fn main() {
    let mut record = vec![];
    let array = array![1.].into_dyn();

    let test = TensorView::from_array(array.view());

    let tensor = TensorView::from_array(array.view());
    tensor.add(&test, &mut record).unwrap();
}

pub fn neg<F>(arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Neg<Output = F> + Clone,
{
    arr.mapv(|x| -x)
}
