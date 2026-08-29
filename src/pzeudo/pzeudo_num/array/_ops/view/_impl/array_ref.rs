use crate::prelude::*;

impl<F, T> OpsSlice<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}
impl<F, T> OpsBroadcast<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}

impl<F, T> OpsBroadcast<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayTrait<F>,
{
}

impl<F, T> OpsPermute<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}
impl<F, T> OpsToShape<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}
