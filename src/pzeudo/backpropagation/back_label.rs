use std::sync::{Arc, RwLock};

use crate::{NDArrayF64, TensorInner};

pub enum BackLabel<'a>
where
    TensorInner<'a>: Sized,
{
    Add(Arc<TensorInner<'a>>, Arc<TensorInner<'a>>),
    Sub(Arc<TensorInner<'a>>, Arc<TensorInner<'a>>),
    Mul(Arc<TensorInner<'a>>, Arc<TensorInner<'a>>),
    Div(Arc<TensorInner<'a>>, Arc<TensorInner<'a>>),
}
