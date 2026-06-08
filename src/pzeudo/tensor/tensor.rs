use std::sync::{Arc, RwLock};

use crate::tensor::backend::PzeudoBackend;

pub struct Tensor<B: PzeudoBackend> {
    inner: Arc<RwLock<B>>,
}

impl<B: PzeudoBackend> Tensor<B> {
    pub fn new(inner: B) -> Tensor<B> {
        Tensor {
            inner: Arc::new(RwLock::new(inner)),
        }
    }
}
