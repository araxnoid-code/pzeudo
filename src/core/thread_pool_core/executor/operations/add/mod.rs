use crate::tensor_core::{ExecuteOps, ExecuteOutput, InnerTensor, Tensor, TensorCore};

pub fn add_execute(a: &InnerTensor, b: &InnerTensor) -> ExecuteOutput {
    ExecuteOutput(Ok(a.add(b)))
}
