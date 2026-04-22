use std::{rc::Rc, sync::Arc};

use cahotic::TaskTrait;

use crate::{
    add_execute, div_execute, mul_execute, sub_execute,
    tensor_core::{InnerTensor, Tensor, execute::ExecuteOutput},
};

// Execute
pub enum ExecuteOps {
    Add(Arc<InnerTensor>, Arc<InnerTensor>),
    Sub(Arc<InnerTensor>, Arc<InnerTensor>),
    Mul(Arc<InnerTensor>, Arc<InnerTensor>),
    Div(Arc<InnerTensor>, Arc<InnerTensor>),
}

impl<'a> TaskTrait<ExecuteOutput> for ExecuteOps {
    fn execute(&self) -> ExecuteOutput {
        match self {
            ExecuteOps::Add(a, b) => add_execute(a, b),
            ExecuteOps::Sub(a, b) => sub_execute(a, b),
            ExecuteOps::Mul(a, b) => mul_execute(a, b),
            ExecuteOps::Div(a, b) => div_execute(a, b),
        }
    }
}
