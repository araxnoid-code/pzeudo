use crate::{ThreadPoolCore, pzeudo_core::core::PzeudoCore};

/// new
impl<const N: usize, const MAX_RING_BUFFER: usize> PzeudoCore<N, MAX_RING_BUFFER> {
    pub fn new() -> Self {
        Self {
            tensor_core: Default::default(),
            thread_pool: Default::default(),
        }
    }
}
