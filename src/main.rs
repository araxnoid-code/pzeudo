use std::{
    borrow,
    cell::RefCell,
    ops::{Add, AddAssign},
    rc::Rc,
};

use ndarray::{ArrayD, ArrayRefD, ArrayViewD, CowArray, Dim, IxDynImpl, array, linalg::Dot};
use pzeudo::Tensor;

fn main() {
    let array: ArrayD<f32> = array![[1., 2., 4.]].into_dyn();
    // let gradient = array![[1., 2., 4.]].into_dyn();

    let tensor = Tensor::from_array(array);
}
