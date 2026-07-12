use crate::{Array, OpsAdd, OpsDiv, OpsMul, OpsSub};

impl<F> OpsAdd<F> for Array<F> where F: Copy {}
impl<F> OpsSub<F> for Array<F> where F: Copy {}
impl<F> OpsDiv<F> for Array<F> where F: Copy {}
impl<F> OpsMul<F> for Array<F> where F: Copy {}
