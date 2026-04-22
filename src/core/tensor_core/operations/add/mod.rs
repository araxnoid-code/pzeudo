use crate::tensor_core::{ExecuteOps, InnerTensor, Tensor, TensorCore, execute::ExecuteOutput};

impl TensorCore {
    pub fn add_execute(&self, a: &Tensor, b: &Tensor) {
        ExecuteOps::Add(a.inner.clone(), b.inner.clone());
    }
}
