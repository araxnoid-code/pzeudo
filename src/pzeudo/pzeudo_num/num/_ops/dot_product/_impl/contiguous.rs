use crate::prelude::*;

impl<F> OpsDotProduct<F> for Array<F> where F: Copy {}
impl OpsDotProductF32 for Array<f32> {}
impl OpsDotProductF64 for Array<f64> {}
