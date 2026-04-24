use cahotic::{PollWaiting, Schedule};

use crate::tensor_core::{ExecuteOps, ExecuteOut};

/// InnerTensorType, functions to differentiate tensors created manually (initial) and tensors obtained through operations (schedule)
pub(crate) enum InnerTensorType {
    Initial(InnerTensor),
    Schedule(InnerTensor),
}

impl InnerTensorType {
    pub fn get_initial_value(&mut self) -> Option<f64> {
        match self {
            Self::Initial(value) => {
                value.poll_to_value();
                value.get_value()
            }
            Self::Schedule(_) => None,
        }
    }
}

/// TensorValue. functions as a value store based on the stages passed by the tensor,
/// if the tensor has just been created through an operation then it will have a value of TensorValue::None,
/// if it has passed the running stage it will change to TensorValue::Poll,
/// if the tensor value already exists then it will change to TensorValue::Value
pub enum TensorValue {
    None,
    Poll(PollWaiting<ExecuteOut>),
    Value(f64),
}

/// InnerTensor. the main structure of a tensor, where all operations occur
pub struct InnerTensor {
    pub(crate) schedule: Option<Schedule<ExecuteOps, ExecuteOps, ExecuteOut>>,
    pub(crate) data: TensorValue,
}

impl InnerTensor {
    pub fn new_from_schedule(
        schedule: Schedule<ExecuteOps, ExecuteOps, ExecuteOut>,
    ) -> InnerTensor {
        InnerTensor {
            schedule: Some(schedule),
            data: TensorValue::None,
        }
    }

    pub fn new_from_value(value: f64) -> InnerTensor {
        InnerTensor {
            schedule: None,
            data: TensorValue::Value(value),
        }
    }

    pub fn get_value(&self) -> Option<f64> {
        if let TensorValue::Value(value) = &self.data {
            Some(*value)
        } else {
            None
        }
    }

    pub fn poll_to_value(&mut self) {
        if let TensorValue::Poll(poll) = &self.data {
            if let Some(value) = poll.get() {
                self.data = TensorValue::Value(value.0.unwrap());
            } else {
                panic!("Error")
            }
        } else {
        }
    }
}
