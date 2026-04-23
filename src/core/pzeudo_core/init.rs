use crate::{ThreadPoolCore, pzeudo_core::core::PzeudoCore};

/// default
impl<const N: usize, const MAX_RING_BUFFER: usize> Default for PzeudoCore<N, MAX_RING_BUFFER> {
    fn default() -> Self {
        Self {
            tensor_core: Default::default(),
            thread_pool: Default::default(),
        }
    }
}
