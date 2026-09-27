use crate::prelude::*;

pub trait DotProductF32F64<F, T> {
    fn dot_f32f64<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        Rhs: ArrayTrait<F>;
}

impl<A, T> DotProductF32F64<f32, T> for A
where
    A: OpsDotProductF32,
{
    fn dot_f32f64<Rhs>(&self, rhs: &Rhs) -> Result<Array<f32>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f32, T>: ArrayTrait<f32>,
        Rhs: ArrayTrait<f32>,
    {
        self.dot_f32(rhs)
    }
}

impl<A, T> DotProductF32F64<f64, T> for A
where
    A: OpsDotProductF64,
{
    fn dot_f32f64<Rhs>(&self, rhs: &Rhs) -> Result<Array<f64>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f64, T>: ArrayTrait<f64>,
        Rhs: ArrayTrait<f64>,
    {
        self.dot_f64(rhs)
    }
}
