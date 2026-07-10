use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, ArrayViewD, array};
use pzeudo::{Module, OpsLabel, Tensor, ndarray_backend};

// impl Float for f16 {}

fn main() {
    let module = Module::new(ndarray_backend::<f32>());

    let array = array![[10.]].into_dyn();

    let array = module.new_tensor(array, None, None).unwrap();
}
