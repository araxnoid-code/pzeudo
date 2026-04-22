use crate::tensor_core::{ExecuteOutput, InnerTensor, Tensor};

pub fn sub_execute(a: &InnerTensor, b: &InnerTensor) -> ExecuteOutput {
    ExecuteOutput(Ok(a.sub(b)))
}
