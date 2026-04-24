use std::{cell::RefCell, sync::Arc};

use cahotic::{Cahotic, DefaultSchedule};

use crate::tensor_core::{ExecuteOps, ExecuteOut, InnerTensorType};

pub struct ThreadPoolCore<const N: usize, const MAX_RING_BUFFER: usize> {
    pub(crate) executor: Cahotic<ExecuteOps, ExecuteOps, ExecuteOut, N, MAX_RING_BUFFER>,
    pub(crate) schedule_order: Vec<Arc<RefCell<InnerTensorType>>>,
}
