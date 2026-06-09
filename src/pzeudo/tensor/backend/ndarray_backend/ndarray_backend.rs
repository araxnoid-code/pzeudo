use std::fmt::Debug;

use crate::{Arr, NDArrayArr, NDArrayDataType, NDArrayGradient, PzeudoBackend};

pub struct NDArrayBackend<T>
where
    T: NDArrayDataType,
{
    pub(crate) arr: NDArrayArr<T>,
    // pub(crate) grad: NDArrayGradient<T>,
}

impl<T> NDArrayBackend<T>
where
    T: NDArrayDataType,
{
    // pub fn new(arr: T) -> NDArrayBackend<T> {
    //     Self { arr }
    // }
}

impl<A, T> PzeudoBackend<A> for NDArrayBackend<T>
where
    A: Arr,
    T: NDArrayDataType,
{
    type ArrType = NDArrayArr<T>;
    //
    fn backend() -> impl Debug {
        "ndarray"
    }

    fn get_arr(&self) -> &Self::ArrType {
        &self.arr
    }

    fn arr_into(arr: Self::ArrType) -> Self {
        Self { arr }
    }
}
