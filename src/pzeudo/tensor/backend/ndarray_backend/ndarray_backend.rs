use std::fmt::Debug;

use crate::{Arr, NDArrayArr, NDArrayDataType, PzeudoBackend, PzeudoDataType};

pub struct NDArrayBackend<T>
where
    T: NDArrayDataType<ScalarType = PzeudoDataType>,
{
    pub(crate) arr: NDArrayArr<T>,
    pub(crate) grad: Option<NDArrayArr<T>>,
}

impl<T> NDArrayBackend<T>
where
    T: NDArrayDataType<ScalarType = PzeudoDataType>,
{
    pub fn new(arr: NDArrayArr<T>, grad: Option<NDArrayArr<T>>) -> NDArrayBackend<T> {
        Self { arr, grad }
    }
}

impl<'s, A, T> PzeudoBackend<'s, A> for NDArrayBackend<T>
where
    A: Arr<'s, ScalarType = PzeudoDataType> + 's,
    T: NDArrayDataType<ScalarType = PzeudoDataType>,
{
    type BackendArrType = NDArrayArr<T>;

    fn backend() -> impl Debug {
        "ndarray"
    }

    fn get_backend_arr(&self) -> &Self::BackendArrType {
        &self.arr
    }

    fn get_backend_grad_as_mut(&mut self) -> &mut Option<Self::BackendArrType> {
        &mut self.grad
    }

    fn arr_into(arr: Self::BackendArrType, grad: bool) -> Self {
        let grad = if grad {
            Some(Self::BackendArrType::zeros(arr.get_shape()))
        } else {
            None
        };

        Self::new(arr, grad)
    }
}
