use cahotic::Cahotic;

use crate::tensor_core::{ExecuteOps, ExecuteOut, InnerTensorType, Tensor, TensorCore};

impl TensorCore {
    pub fn add_execute(&self, a: &Tensor, b: &Tensor) -> ExecuteOps {
        let execute_ops = ExecuteOps::AddExecute(
            a.inner.borrow_mut().get_initial_value(),
            b.inner.borrow_mut().get_initial_value(),
        );

        execute_ops
    }
}
