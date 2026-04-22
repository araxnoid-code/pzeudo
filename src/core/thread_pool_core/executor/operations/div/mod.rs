use crate::tensor_core::{ExecuteOutput, InnerTensor, Tensor};

pub fn div_execute(a: &InnerTensor, b: &InnerTensor) -> ExecuteOutput {
    ExecuteOutput(Ok(a.div(b)))
}
