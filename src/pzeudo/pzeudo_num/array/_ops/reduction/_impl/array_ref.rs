use crate::prelude::*;

impl<F, T> OpsSum<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}
impl<F, T> OpsAvg<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}
