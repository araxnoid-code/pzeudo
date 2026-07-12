use crate::{Array, OpsMatmul2DF32, OpsMatmul2DF64, OpsMatmulNDF32, OpsMatmulNDF64};

impl OpsMatmul2DF32 for Array<f32> {}
impl OpsMatmul2DF64 for Array<f64> {}

impl OpsMatmulNDF32 for Array<f32> {}
impl OpsMatmulNDF64 for Array<f64> {}
