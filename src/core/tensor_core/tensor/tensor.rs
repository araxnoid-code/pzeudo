use std::{cell::RefCell, sync::Arc};

use cahotic::Schedule;

use crate::tensor_core::{ExecuteOps, ExecuteOut, InnerTensorType};

pub struct Tensor {
    pub(crate) inner: Arc<RefCell<InnerTensorType>>,
}
