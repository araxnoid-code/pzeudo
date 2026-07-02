use ndarray::{ArrayD, ArrayViewD, Axis};

use crate::{OwnAble, RefAble, TensorAble};

pub trait OpsAble {
    type TensorType: TensorAble;
    type RefTensorType<'a>: TensorAble
    where
        Self: 'a;

    fn add(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized;

    fn sub(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized;

    fn mul(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized;

    fn div(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized;

    // axis ops
    fn ops_sum_axis(&self, axis: usize) -> Self::TensorType;
    fn ops_insert_axis(&self, axis: usize) -> Self::RefTensorType<'_>;
}

impl OpsAble for ArrayViewD<'_, f32> {
    type TensorType = ArrayD<f32>;
    type RefTensorType<'a>
        = ArrayViewD<'a, f32>
    where
        Self: 'a;

    fn add(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized,
    {
        self + rhs
    }

    fn sub(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized,
    {
        self - rhs
    }

    fn mul(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized,
    {
        self * rhs
    }

    fn div(&self, rhs: &Self) -> Self::TensorType
    where
        Self: Sized,
    {
        self / rhs
    }

    fn ops_sum_axis(&self, axis: usize) -> Self::TensorType {
        self.sum_axis(Axis(axis))
    }

    fn ops_insert_axis(&self, axis: usize) -> Self::RefTensorType<'_> {
        self.view().insert_axis(Axis(axis))
    }
}
