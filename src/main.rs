use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, CowArray};
use pzeudo::{GradStorage, Tensor};

fn main() {
    let array = ArrayD::<f32>::zeros(vec![10, 10]);
    let array_view = array.view();
    let array_cow = CowArray::from(&array);

    let grad_storage = Rc::new(RefCell::new(GradStorage::new(None)));
    // let tensor = Tensor::new(array, None, grad_storage.clone()).unwrap();
    let tensor_view = Tensor::new(array_view, None, grad_storage.clone()).unwrap();
    let tensor_cow = Tensor::new(array_cow, None, grad_storage).unwrap();
}
