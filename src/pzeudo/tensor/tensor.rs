use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{Arr, tensor::backend_conf::PzeudoBackend};

pub struct Tensor<A: Arr, B: PzeudoBackend<A>> {
    pub(crate) inner: Arc<RwLock<B>>,
    _phantom: PhantomData<A>,
    // pub(crate) label: Option<BackwardLabel<B>>,
}

impl<A, B> Tensor<A, B>
where
    A: Arr,
    B: PzeudoBackend<A>,
{
    pub fn new(inner: B) -> Tensor<A, B> {
        Tensor {
            inner: Arc::new(RwLock::new(inner)),
            _phantom: Default::default(),
        }
    }
}
