use std::sync::{Arc, RwLock};

use crate::{Arr, PzeudoBackend};

pub enum BackwardLabel<A, B>
where
    A: Arr,
    B: PzeudoBackend<A>,
{
    Add(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Sub(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Mul(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Div(Arc<RwLock<B>>, Arc<RwLock<B>>),
    _Phantom(A),
}
