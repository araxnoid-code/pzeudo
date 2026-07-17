use std::ops::{Add, Div, Mul, Sub};

use crate::{Array, ArrayTrait, OpsAdd, OpsDiv, OpsMul, OpsSub};

impl<F> OpsAdd<F> for Array<F> where F: Copy {}
impl<F> OpsAdd<F> for &Array<F> where F: Copy {}
impl<Rhs, F> Add<Rhs> for Array<F>
where
    Rhs: ArrayTrait<F>,
    F: Add<Output = F> + Copy,
{
    type Output = Array<F>;

    fn add(self, rhs: Rhs) -> Self::Output {
        OpsAdd::add(&self, &rhs).unwrap()
    }
}

impl<F> OpsSub<F> for Array<F> where F: Copy {}
impl<Rhs, F> Sub<Rhs> for Array<F>
where
    Rhs: ArrayTrait<F>,
    F: Sub<Output = F> + Copy,
{
    type Output = Array<F>;

    fn sub(self, rhs: Rhs) -> Self::Output {
        OpsSub::sub(&self, &rhs).unwrap()
    }
}

impl<F> OpsDiv<F> for Array<F> where F: Copy {}
impl<Rhs, F> Div<Rhs> for Array<F>
where
    Rhs: ArrayTrait<F>,
    F: Div<Output = F> + Copy,
{
    type Output = Array<F>;

    fn div(self, rhs: Rhs) -> Self::Output {
        OpsDiv::div(&self, &rhs).unwrap()
    }
}

impl<F> OpsMul<F> for Array<F> where F: Copy {}
impl<Rhs, F> Mul<Rhs> for Array<F>
where
    Rhs: ArrayTrait<F>,
    F: Mul<Output = F> + Copy,
{
    type Output = Array<F>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        OpsMul::mul(&self, &rhs).unwrap()
    }
}
