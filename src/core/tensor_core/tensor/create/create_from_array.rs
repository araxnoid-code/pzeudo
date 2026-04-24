use std::{cell::RefCell, sync::Arc};

use crate::{
    TensorCore,
    tensor_core::{InnerTensorType, Tensor},
};

impl TensorCore {
    pub fn create_tensor_from_value(&self, value: f64) -> Tensor {
        Tensor::init(value)
    }
}
