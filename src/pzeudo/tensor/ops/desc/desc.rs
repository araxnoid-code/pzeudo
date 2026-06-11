use std::sync::RwLockReadGuard;

use crate::{Arr, PzeudoBackend, ShapeTrait, Tensor};

impl<'a, A, B> Tensor<'a, A, B>
where
    A: Arr<'a>,
    B: PzeudoBackend<'a, A>,
{
    pub fn read_inner(&'a self) {}
}
