use std::{cell::RefCell, sync::Arc};

use cahotic::Schedule;

use crate::{ExecuteOps, ExecuteOut, InnerTensor, InnerTensorType, Tensor};

/// Tensor. structure that wraps InnerTensor, functions as an abstraction of operations on innerTensor
impl Tensor {
    pub fn init(value: f64) -> Tensor {
        Tensor {
            inner: Arc::new(RefCell::new(InnerTensorType::Initial(
                InnerTensor::new_from_value(value),
            ))),
        }
    }

    pub(crate) fn sch(schedule: Schedule<ExecuteOps, ExecuteOps, ExecuteOut>) -> Tensor {
        Tensor {
            inner: Arc::new(RefCell::new(InnerTensorType::Schedule(
                InnerTensor::new_from_schedule(schedule),
            ))),
        }
    }
}
