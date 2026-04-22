use cahotic::TaskTrait;

use crate::tensor_core::{add, div, execution::ExecutionOutput, mul, sub};

// Execution
pub enum ExecutionOps {
    Add(i32, i32),
    Sub(i32, i32),
    Mul(i32, i32),
    Div(i32, i32),
}

impl TaskTrait<ExecutionOutput> for ExecutionOps {
    fn execute(&self) -> ExecutionOutput {
        match self {
            ExecutionOps::Add(a, b) => add(*a, *b),
            ExecutionOps::Sub(a, b) => sub(*a, *b),
            ExecutionOps::Mul(a, b) => mul(*a, *b),
            ExecutionOps::Div(a, b) => div(*a, *b),
        }
    }
}
