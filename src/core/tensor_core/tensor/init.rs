use std::{cell::RefCell, sync::Arc};

use cahotic::Schedule;

use crate::{ExecuteOps, ExecuteOut, InnerTensor, Tensor};

impl Tensor {
    pub fn init(value: f64) -> Tensor {
        Tensor {
            inner: Arc::new(RefCell::new(InnerTensor::Initial(value))),
        }
    }

    pub(crate) fn sch(schedule: Schedule<ExecuteOps, ExecuteOps, ExecuteOut>) -> Tensor {
        Tensor {
            inner: Arc::new(RefCell::new(InnerTensor::Schedule(Some(schedule)))),
        }
    }
}
