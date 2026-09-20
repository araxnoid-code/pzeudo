use crate::prelude::*;

impl<F> OpsDotProduct<F> for ArrayView<'_, F> where F: Copy {}
impl OpsDotProductF32 for ArrayView<'_, f32> {}
impl OpsDotProductF64 for ArrayView<'_, f64> {}
