use cahotic::PollWaiting;

use crate::{
    tensor_core::{ExecuteOps, ExecuteOutput},
    thread_pool_core::ThreadPoolCore,
};

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    pub fn executor(&self, execution: ExecuteOps) -> PollWaiting<ExecuteOutput> {
        self.executor.spawn_task(execution)
    }
}
