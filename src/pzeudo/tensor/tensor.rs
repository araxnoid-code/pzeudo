use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    Arr, PzeudoDataType,
    tensor::{backend_conf::PzeudoBackend, ops::BackwardLabel},
};

pub struct Tensor<'s, A: Arr<'s>, B: PzeudoBackend<'s, A>> {
    pub(crate) inner: Arc<RwLock<B>>,
    pub(crate) label: Option<BackwardLabel<'s, A, B>>,
}

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s>,
    B: PzeudoBackend<'s, A>,
{
    pub fn new(inner: B, label: Option<BackwardLabel<'s, A, B>>) -> Tensor<'s, A, B> {
        Tensor {
            inner: Arc::new(RwLock::new(inner)),
            label,
        }
    }

    pub fn zeros() {}
}
