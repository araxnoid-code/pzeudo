use std::{fmt::Debug, marker::PhantomData};

use crate::{Arr, PzeudoBackend, PzeudoDataType, tensor::backend_conf};

pub struct NDArrayBackend<A>
where
    A: Arr,
{
    pub(crate) inner: A,
    pub(crate) grad: Option<A>,
}

impl<A> NDArrayBackend<A>
where
    A: Arr,
{
    pub fn new(inner: A, grad: Option<A>) -> NDArrayBackend<A> {
        Self { inner, grad }
    }
}

impl<A> PzeudoBackend<A> for NDArrayBackend<A>
where
    A: Arr<ScalarType = PzeudoDataType>,
{
    type ShapeType = A::ShapeType;

    fn arr_into(arr: A, grad: bool) -> Self {
        let shape = arr.get_shape();

        Self {
            inner: arr,
            grad: if grad { Some(A::zeros(shape)) } else { None },
        }
    }

    // desc
    fn get_array(&self) -> &A::InnerArrType {
        self.inner.get_array()
    }

    fn get_grad(&self) -> Option<&A::InnerArrType> {
        if let Some(grad) = self.grad.as_ref() {
            Some(grad.get_array())
        } else {
            None
        }
    }

    fn get_shape(&self) -> Self::ShapeType {
        self.inner.get_shape()
    }

    // innner getter
    fn backend() -> impl Debug {
        "ndarray"
    }

    fn get_backend_arr(&self) -> &A {
        &self.inner
    }

    fn get_backend_grad_as_mut(&mut self) -> &mut Option<A> {
        &mut self.grad
    }
}
