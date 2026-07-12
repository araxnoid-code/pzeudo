use crate::{ArrayView, OpsMatmul2DF32, OpsMatmul2DF64};

impl OpsMatmul2DF32 for ArrayView<'_, f32> {}
impl OpsMatmul2DF64 for ArrayView<'_, f64> {}
