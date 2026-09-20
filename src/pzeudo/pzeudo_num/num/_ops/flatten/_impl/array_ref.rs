use crate::{ArrayRef, ArrayRefMut, ArrayTrait, OpsFlatten};

impl<F, T> OpsFlatten<F> for ArrayRef<'_, F, T> where for<'a> ArrayRef<'a, F, T>: ArrayTrait<F> {}

impl<F, T> OpsFlatten<F> for ArrayRefMut<'_, F, T> where for<'a> ArrayRefMut<'a, F, T>: ArrayTrait<F>
{}
