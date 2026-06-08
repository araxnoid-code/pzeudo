use std::sync::{Arc, RwLock};

use crate::{PzeudoBackend, Tensor};

pub enum BackwardLabel<B>
where
    B: PzeudoBackend,
{
    Add(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Sub(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Mul(Arc<RwLock<B>>, Arc<RwLock<B>>),
    Div(Arc<RwLock<B>>, Arc<RwLock<B>>),
}
