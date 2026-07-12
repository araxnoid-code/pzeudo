use crate::{ArrayView, OpsAdd, OpsDiv, OpsMul, OpsSub};

impl<F> OpsAdd<F> for ArrayView<'_, F> where F: Copy {}
impl<F> OpsSub<F> for ArrayView<'_, F> where F: Copy {}
impl<F> OpsDiv<F> for ArrayView<'_, F> where F: Copy {}
impl<F> OpsMul<F> for ArrayView<'_, F> where F: Copy {}
