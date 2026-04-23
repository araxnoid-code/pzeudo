use cahotic::Cahotic;

use crate::tensor_core::{ExecuteOps, ExecuteOut, InnerTensor, Tensor, TensorCore};

impl TensorCore {
    pub fn add_execute(&self, a: &Tensor, b: &Tensor) -> ExecuteOps {
        let execute_ops = ExecuteOps::AddExecute(
            a.inner.borrow().get_initial(),
            b.inner.borrow().get_initial(),
        );

        execute_ops
    }
}
