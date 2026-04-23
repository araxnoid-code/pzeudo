use cahotic::{DefaultOutput, ScheduleVec, SchedulerTrait, TaskTrait};

use crate::{add_executor, tensor_core::ExecuteOut};

pub(crate) enum ExecuteOps {
    AddExecute(Option<f64>, Option<f64>),
    SubExecute(Option<f64>, Option<f64>),
}

impl SchedulerTrait<ExecuteOut> for ExecuteOps {
    fn execute(&self, scheduler_vec: ScheduleVec<ExecuteOut>) -> ExecuteOut {
        match self {
            Self::AddExecute(a, b) => add_executor(a, b, Some(scheduler_vec)),
            Self::SubExecute(a, b) => ExecuteOut(Err(())),
        }
    }
}

impl TaskTrait<ExecuteOut> for ExecuteOps {
    fn execute(&self) -> ExecuteOut {
        match self {
            Self::AddExecute(a, b) => add_executor(a, b, None),
            ExecuteOps::SubExecute(a, b) => ExecuteOut(Err(())),
        }
    }
}
