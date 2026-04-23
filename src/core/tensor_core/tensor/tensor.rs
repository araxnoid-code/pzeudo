use std::{cell::RefCell, sync::Arc};

use cahotic::Schedule;

use crate::tensor_core::{ExecuteOps, ExecuteOut, InnerTensor};

pub struct Tensor {
    pub(crate) inner: Arc<RefCell<InnerTensor>>,
}
