use std::{cell::RefCell, sync::Arc};

use cahotic::Cahotic;

use crate::{
    ThreadPoolCore,
    tensor_core::{ExecuteOps, ExecuteOut, InnerTensor},
};

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    pub fn get_executor(&self) -> &Cahotic<ExecuteOps, ExecuteOps, ExecuteOut, N, MAX_RING_BUFFER> {
        &self.executor
    }

    pub fn get_schedule_order(&self) -> &Vec<Arc<RefCell<InnerTensor>>> {
        &self.schedule_order
    }
}
