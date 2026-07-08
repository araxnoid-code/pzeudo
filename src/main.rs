use std::{cell::RefCell, rc::Rc};

use ndarray::array;
use pzeudo::{BackwardTrait, GradStorage, Tensor};

fn main() {
    let grad_storage = Rc::new(RefCell::new(GradStorage::new(None)));
    let record_storage = Rc::new(RefCell::new(vec![]));
    let tensor_a = Tensor::from_array(
        array![10.].into_dyn(),
        grad_storage.clone(),
        record_storage.clone(),
    )
    .unwrap();

    let array = array![10.].into_dyn();
    let tensor_b =
        Tensor::from_array(array.view(), grad_storage.clone(), record_storage.clone()).unwrap();

    let tensor = tensor_b.matmul_2d(&tensor_a).unwrap();
    tensor.backward().unwrap();
}
