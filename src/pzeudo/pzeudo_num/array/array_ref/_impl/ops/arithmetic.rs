use std::ops::{Add, Div, Mul, Sub};

use crate::{Array, ArrayRef, ArrayTrait, OpsAdd, OpsDiv, OpsMul, OpsSub};

impl<F, T> OpsAdd<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}

impl<F, T> OpsMul<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}

impl<F, T> OpsDiv<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}

impl<F, T> OpsSub<F> for ArrayRef<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
}
