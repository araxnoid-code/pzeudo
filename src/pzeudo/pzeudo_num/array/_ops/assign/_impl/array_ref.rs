use crate::prelude::*;

impl<F, T> ArrayAssignTrait<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayTrait<F>,
{
    fn get_mut_metadata(&mut self) -> MutMetadata<'_, F> {
        MutMetadata {
            data: self.data,
            offset: self.offset,
            stride: self.stride,
            shape: self.shape,
        }
    }
}

impl<F, T> OpsAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsAddAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsDivAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsMulAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsSubAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}
