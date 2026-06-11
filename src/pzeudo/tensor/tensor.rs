use std::{
    fmt::Display,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    Arr, PzeudoDataType,
    tensor::{backend_conf::PzeudoBackend, ops::BackwardLabel},
};

pub struct Tensor<'s, A: Arr<'s>, B: PzeudoBackend<'s, A>> {
    pub(crate) backend: B,
    pub(crate) label: Option<BackwardLabel<'s, A, B>>,
}

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s>,
    B: PzeudoBackend<'s, A>,
{
    pub fn new(backend: B, label: Option<BackwardLabel<'s, A, B>>) -> Tensor<'s, A, B> {
        Tensor { backend, label }
    }

    pub fn zeros() {}
}
