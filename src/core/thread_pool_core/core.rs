use cahotic::{Cahotic, DefaultSchedule};

use crate::tensor_core::{ExecutionOps, ExecutionOutput};

pub struct ThreadPoolCore<const N: usize, const MAX_RING_BUFFER: usize> {
    pub(crate) executor: Cahotic<
        ExecutionOps,
        DefaultSchedule<ExecutionOutput>,
        ExecutionOutput,
        N,
        MAX_RING_BUFFER,
    >,
}
