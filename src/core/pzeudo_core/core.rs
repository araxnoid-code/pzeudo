use crate::{tensor_core::TensorCore, thread_pool_core::ThreadPoolCore};

pub struct PzeudoCore<const N: usize, const MAX_RING_BUFFER: usize> {
    pub(crate) thread_pool: ThreadPoolCore<N, MAX_RING_BUFFER>,
    pub(crate) tensor_core: TensorCore,
}
