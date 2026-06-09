use std::sync::{Arc, RwLock};

use crate::{Arr, PzeudoBackend};

pub enum BackwardLabel<'s, A, B>
where
    A: Arr<'s>,
    B: PzeudoBackend<'s, A>,
{
    Add(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Sub(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Mul(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Div(Arc<RwLock<B>>, Arc<RwLock<B>>),
    _Phantom(&'s A),
}
