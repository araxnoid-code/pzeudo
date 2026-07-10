use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, ArrayViewD, array};
use pzeudo::{Module, OpsLabel, Tensor, ndarray_backend};

// impl Float for f16 {}

fn main() {
    let data = Data {
        data: String::from("halo"),
    };
}

struct Data {
    data: String,
}

struct DataView<'a> {
    view: &'a String,
}
