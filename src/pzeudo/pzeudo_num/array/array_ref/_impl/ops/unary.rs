use crate::prelude::*;

impl<F, T> OpsUnary<F> for ArrayRef<'_, F, T> where for<'a> ArrayRef<'a, F, T>: ArrayTrait<F> {}
