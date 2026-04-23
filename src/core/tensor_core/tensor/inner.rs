use cahotic::{DefaultOutput, Schedule};

use crate::tensor_core::{ExecuteOps, ExecuteOut};

pub(crate) enum InnerTensor {
    Initial(f64),
    Schedule(Option<Schedule<ExecuteOps, ExecuteOps, ExecuteOut>>),
}

impl InnerTensor {
    pub fn get_initial(&self) -> Option<f64> {
        match self {
            Self::Initial(value) => Some(*value),
            Self::Schedule(_) => None,
        }
    }
}
