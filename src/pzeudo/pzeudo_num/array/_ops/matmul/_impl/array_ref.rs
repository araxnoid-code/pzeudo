use crate::prelude::*;

impl<T> OpsMatmul2DF32 for ArrayRef<'_, f32, T> where for<'a> ArrayRef<'a, f32, T>: ArrayTrait<f32> {}
impl<T> OpsMatmul2DF64 for ArrayRef<'_, f64, T> where for<'a> ArrayRef<'a, f64, T>: ArrayTrait<f64> {}
