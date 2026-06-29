use std::{
    ops::Add,
    sync::{Arc, RwLock},
};

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr, array};
use pzeudo::{Backward, Tensor, TensorTrait};

fn main() {
    let mut record = vec![];
    let tensor_a = Tensor::from_array(array![5.].into_dyn());
    let tensor_b = Tensor::from_array(array![2.].into_dyn());
    let tensor_c = Tensor::from_array(array![4.].into_dyn());
    // let tensor_d = Tensor::from_array(array![8.].into_dyn());
    // let tensor_f = Tensor::from_array(array![1.].into_dyn());

    let res_a = tensor_a.add(&tensor_b, &mut record);
    let res_b = res_a.mul(&tensor_c, &mut record);
    res_b.backward(&record);

    let shared_grad = tensor_a.get_share_gradient().unwrap();
    let grad = shared_grad.borrow();
    println!("{}", grad);
}
