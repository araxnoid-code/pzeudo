use crate::tensor_core::{ExecuteOutput, InnerTensor, Tensor};

pub fn mul_execute(a: &InnerTensor, b: &InnerTensor) -> ExecuteOutput {
    ExecuteOutput(Ok(a.mul(b)))
}
