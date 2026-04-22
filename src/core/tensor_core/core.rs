use crate::{
    tensor_core::{GraphType, Tensor},
    thread_pool_core::ThreadPoolCore,
};

pub struct TensorCore {}

/// Create
impl TensorCore {
    /// create_tensor(&self, value: f64);
    /// Fungi that play a role in early development
    pub fn create_tensor(&self, value: f64) -> Tensor {
        Tensor::init(value, GraphType::Initial)
    }
}
