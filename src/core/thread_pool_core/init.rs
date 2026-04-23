use cahotic::{Cahotic, CahoticBuilder};

use crate::ThreadPoolCore;

impl<const N: usize, const MAX_RING_BUFFER: usize> Default for ThreadPoolCore<N, MAX_RING_BUFFER> {
    fn default() -> Self {
        Self {
            executor: Cahotic::init().unwrap(),
            schedule_order: vec![],
        }
    }
}
