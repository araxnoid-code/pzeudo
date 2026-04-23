use crate::PzeudoCore;

/// executor, wrapper for ThreadPoolCore, intended for functions that have a relationship to the executor stage
impl<const N: usize, const MAX_RING_BUFFER: usize> PzeudoCore<N, MAX_RING_BUFFER> {
    /// wrapper of Thread Pool::executor_schedule
    pub fn run(&mut self) {
        self.thread_pool.execute_schedule();
    }

    /// functions as the end of the executor and the pzeudo itself
    pub fn join(self) {
        self.thread_pool.executor.join();
    }
}
