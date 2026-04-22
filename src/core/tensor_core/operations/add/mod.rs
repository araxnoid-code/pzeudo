use crate::tensor_core::execution::ExecutionOutput;

pub fn add(a: i32, b: i32) -> ExecutionOutput {
    ExecutionOutput(Ok(a + b))
}
