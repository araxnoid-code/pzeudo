use crate::prelude::*;

impl<F> OpsSum<F> for ArrayView<'_, F> where for<'a> F: Copy {}
impl<F> OpsAvg<F> for ArrayView<'_, F> where for<'a> F: Copy {}
