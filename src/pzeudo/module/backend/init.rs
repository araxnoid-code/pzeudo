use std::marker::PhantomData;

use num_traits::Float;

use crate::{GradientStorage, OpsLabel};

/// Initial Module
pub struct ModuleInit<GradStorage, Record>(pub GradStorage, pub PhantomData<Record>);

/// Ndarray Backend
pub fn ndarray_backend<'ops_label, F>() -> ModuleInit<GradientStorage<F>, OpsLabel<'ops_label, F>>
where
    F: Float,
{
    ModuleInit(GradientStorage::new(None), Default::default())
}
