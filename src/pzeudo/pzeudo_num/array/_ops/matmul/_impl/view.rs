use crate::prelude::*;

impl OpsMatmul2DF32 for ArrayView<'_, f32> {}
impl OpsMatmul2DF64 for ArrayView<'_, f64> {}

impl OpsMatmulNDF32 for ArrayView<'_, f32> {}
impl OpsMatmulNDF64 for ArrayView<'_, f64> {}
