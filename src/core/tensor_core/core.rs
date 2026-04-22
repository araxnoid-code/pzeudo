use crate::tensor_core::{GraphType, Tensor};

pub struct TensorCore {}

/// Create
impl TensorCore {
    /// create_tensor(&self, value: f64);
    /// Fungi that play a role in early development
    pub fn create_tensor(&self, value: f64) -> Tensor {
        Tensor {
            data: value,
            graph_type: GraphType::Initial,
        }
    }
}
