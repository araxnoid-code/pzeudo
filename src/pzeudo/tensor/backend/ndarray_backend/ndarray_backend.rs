use std::fmt::Debug;

use crate::{Arr, NDArrayArr, NDArrayDataType, PzeudoBackend};

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

impl<'s, A, T> PzeudoBackend<'s, A> for NDArrayBackend<T>
where
    A: Arr<'s>,
    T: NDArrayDataType,
{
    type ArrType = NDArrayArr<T>;

    fn backend() -> impl Debug {
        "ndarray"
    }

    fn get_arr(&self) -> &Self::ArrType {
        &self.arr
    }

    fn arr_into(arr: Self::ArrType, grad: bool) -> Self {
        let grad = if grad {
            Some(Self::ArrType::zeros(arr.get_shape()))
        } else {
            None
        };

        Self::new(arr, grad)
    }
}
