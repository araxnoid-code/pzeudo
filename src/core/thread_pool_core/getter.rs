use crate::thread_pool_core::ThreadPoolCore;

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    pub fn get_executor(
        &self,
    ) -> &cahotic::Cahotic<
        crate::tensor_core::ExecutionOps,
        cahotic::DefaultSchedule<crate::tensor_core::ExecutionOutput>,
        crate::tensor_core::ExecutionOutput,
        N,
        MAX_RING_BUFFER,
    > {
        &self.executor
    }
}
