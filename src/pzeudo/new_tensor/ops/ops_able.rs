use ndarray::ArrayViewD;

use crate::OwnAble;

pub trait OpsAble {
    fn add(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized;

    fn sub(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized;

    fn mul(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized;

    fn div(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized;
}

impl<'a> OpsAble for ArrayViewD<'a, f32> {
    fn add(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized,
    {
        self + rhs
    }

    fn sub(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized,
    {
        self - rhs
    }

    fn mul(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized,
    {
        self * rhs
    }

    fn div(&self, rhs: &Self) -> impl OwnAble
    where
        Self: Sized,
    {
        self / rhs
    }
}
