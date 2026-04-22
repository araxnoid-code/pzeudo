use cahotic::{Cahotic, DefaultSchedule};

use crate::tensor_core::{ExecuteOps, ExecuteOutput};

pub struct ThreadPoolCore<const N: usize, const MAX_RING_BUFFER: usize> {
    pub(crate) executor:
        Cahotic<ExecuteOps, DefaultSchedule<ExecuteOutput>, ExecuteOutput, N, MAX_RING_BUFFER>,
}
