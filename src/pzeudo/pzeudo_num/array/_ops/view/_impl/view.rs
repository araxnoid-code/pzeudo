use crate::prelude::*;
impl<F> OpsSlicing<F> for ArrayView<'_, F> where F: Copy {}
impl<F> OpsBroadcast<F> for ArrayView<'_, F> where F: Copy {}
impl<F> OpsPermute<F> for ArrayView<'_, F> where F: Copy {}
