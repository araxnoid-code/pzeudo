use std::{
    ops::Add,
    sync::{Arc, RwLock},
};

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr, array};
use pzeudo::Tensor;

fn main() {
    let tensor_a = Tensor::from_array(array![5.].into_dyn());
    let tensor_b = Tensor::from_array(array![2.].into_dyn());
    let tensor_c = Tensor::from_array(array![4.].into_dyn());
    let tensor_d = Tensor::from_array(array![8.].into_dyn());
    let tensor_f = Tensor::from_array(array![1.].into_dyn());
}
