use std::fmt::Debug;

use crate::Arr;

///
pub trait PzeudoBackend<'s, A>
where
    A: Arr<'s>,
{
    //
    type ArrType: Arr<'s>;
    fn backend() -> impl Debug;

    //
    fn get_arr(&self) -> &Self::ArrType;
    fn arr_into(arr: Self::ArrType, grad: bool) -> Self;

    //
    fn add(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_arr();
        let rhs = rhs.get_arr();
        let output = lhs.add(&rhs);

        Self::arr_into(output, true)
    }

    fn sub(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_arr();
        let rhs = rhs.get_arr();
        let output = lhs.sub(&rhs);

        Self::arr_into(output, true)
    }

    fn mul(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_arr();
        let rhs = rhs.get_arr();
        let output = lhs.mul(&rhs);

        Self::arr_into(output, true)
    }

    fn div(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_arr();
        let rhs = rhs.get_arr();
        let output = lhs.div(&rhs);

        Self::arr_into(output, true)
    }
}
