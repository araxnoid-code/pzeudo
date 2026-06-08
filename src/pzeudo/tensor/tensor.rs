use std::sync::{Arc, RwLock};

use crate::tensor::{backend::PzeudoBackend, ops::BackwardLabel};

pub struct Tensor<B: PzeudoBackend> {
    pub(crate) inner: Arc<RwLock<B>>,
    pub(crate) label: Option<BackwardLabel<B>>,
}

impl<B> Tensor<B>
where
    B: PzeudoBackend,
{
    pub fn new(inner: B, label: Option<BackwardLabel<B>>) -> Tensor<B> {
        Tensor {
            inner: Arc::new(RwLock::new(inner)),
            label,
        }
    }
}
