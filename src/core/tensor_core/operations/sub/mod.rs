use crate::tensor_core::execution::ExecutionOutput;

pub fn sub(a: i32, b: i32) -> ExecutionOutput {
    ExecutionOutput(Ok(a - b))
}
