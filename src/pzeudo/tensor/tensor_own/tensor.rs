use std::sync::{Arc, Mutex};

use ndarray::ArrayD;

struct Tensor {
    array: ArrayD<f32>,
    gradient: Option<Arc<Mutex<ArrayD<f32>>>>,
}

impl Tensor {
    pub fn new(array: ArrayD<f32>, gradient: Option<Arc<Mutex<ArrayD<f32>>>>) -> Tensor {
        Self { array, gradient }
    }
}
