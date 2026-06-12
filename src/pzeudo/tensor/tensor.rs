use std::{
    fmt::Display,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    Arr, PzeudoDataType,
    tensor::{backend_conf::PzeudoBackend, ops::BackwardLabel},
};

pub struct Tensor<A: Arr, B: PzeudoBackend<A>> {
    pub(crate) backend: B,
    pub(crate) label: Option<BackwardLabel<A, B>>,
}

impl<A, B> Tensor<A, B>
where
    A: Arr,
    B: PzeudoBackend<A>,
{
    pub fn new(backend: B, label: Option<BackwardLabel<A, B>>) -> Tensor<A, B> {
        Tensor { backend, label }
    }

    pub fn zeros() {}
}
