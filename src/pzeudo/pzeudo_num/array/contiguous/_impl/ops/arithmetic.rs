use crate::{Array, OpsAdd, OpsDiv, OpsMatmul2DF32, OpsMatmul2DF64, OpsMul, OpsSub};

impl<F> OpsAdd<F> for Array<F> where F: Copy {}
impl<F> OpsSub<F> for Array<F> where F: Copy {}
impl<F> OpsDiv<F> for Array<F> where F: Copy {}
impl<F> OpsMul<F> for Array<F> where F: Copy {}
