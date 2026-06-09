use std::fmt::Debug;

use crate::{Arr, NDArrayArr, NDArrayDataType, NDArrayGradient, PzeudoBackend};

pub struct NDArrayBackend<T>
where
    T: NDArrayDataType,
{
    pub(crate) arr: NDArrayArr<T>,
    pub(crate) grad: Option<NDArrayArr<T>>,
}

impl<T> NDArrayBackend<T>
where
    T: NDArrayDataType,
{
    pub fn new(arr: NDArrayArr<T>, grad: Option<NDArrayArr<T>>) -> NDArrayBackend<T> {
        Self { arr, grad }
    }
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

    fn arr_into_no_grad(arr: Self::ArrType) -> Self {
        Self::new(arr, None)
    }
}
