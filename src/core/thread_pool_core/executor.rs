use cahotic::PollWaiting;

use crate::{
    tensor_core::{ExecutionOps, ExecutionOutput},
    thread_pool_core::ThreadPoolCore,
};

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    pub fn execute(&self, execution: ExecutionOps) -> PollWaiting<ExecutionOutput> {
        self.executor.spawn_task(execution)
    }
}
