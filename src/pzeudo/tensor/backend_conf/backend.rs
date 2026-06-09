use std::fmt::Debug;

use crate::Arr;

///
pub trait PzeudoBackend<A>
where
    A: Arr,
{
    type ArrType: Arr;

    fn backend() -> impl Debug;
    fn get_arr(&self) -> &Self::ArrType;

    fn arr_into(arr: Self::ArrType) -> Self;

    fn add(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_arr();
        let rhs = rhs.get_arr();
        let output = lhs.add(&rhs);

        Self::arr_into(output)
    }
}
