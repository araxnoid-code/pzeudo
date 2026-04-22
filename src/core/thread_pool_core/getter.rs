use crate::thread_pool_core::ThreadPoolCore;

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    pub fn get_executor(
        &self,
    ) -> &cahotic::Cahotic<
        crate::tensor_core::ExecuteOps,
        cahotic::DefaultSchedule<crate::tensor_core::ExecuteOutput>,
        crate::tensor_core::ExecuteOutput,
        N,
        MAX_RING_BUFFER,
    > {
        &self.executor
    }
}
