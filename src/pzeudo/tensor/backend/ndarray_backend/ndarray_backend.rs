use std::{fmt::Debug, marker::PhantomData};

use crate::{Arr, PzeudoBackend, PzeudoDataType, tensor::backend_conf};

pub struct NDArrayBackend<'a, A>
where
    A: Arr<'a>,
{
    pub(crate) inner: A,
    pub(crate) grad: Option<A>,
    _phantom: PhantomData<&'a A>,
}

impl<'a, A> NDArrayBackend<'a, A>
where
    A: Arr<'a>,
{
    pub fn new(inner: A, grad: Option<A>) -> NDArrayBackend<'a, A> {
        Self {
            inner,
            grad,
            _phantom: Default::default(),
        }
    }
}

impl<'a, A> PzeudoBackend<'a, A> for NDArrayBackend<'a, A>
where
    A: Arr<'a, ScalarType = PzeudoDataType>,
{
    type ShapeType = A::ShapeType;

    fn arr_into(arr: A, grad: bool) -> Self {
        // let shape = arr.get_shape();
        // let grad = if grad {
        //     // Some(A::zeros())
        // } else {
        //     None
        // };

        Self {
            inner: arr,
            grad: None,
            _phantom: Default::default(),
        }
    }
    // fn arr_into(arr: A, grad: bool) -> Self {
    //     let grad = if grad {
    //         Some(A::zeros(arr.get_shape()))
    //     } else {
    //         None
    //     };

    //     Self::new(arr, grad)
    // }

    // desc
    fn get_array(&'a self) -> &'a A::InnerArrType {
        self.inner.get_array()
    }

    fn get_grad(&'a self) -> Option<&'a A::InnerArrType> {
        if let Some(grad) = self.grad.as_ref() {
            Some(grad.get_array())
        } else {
            None
        }
    }

    fn get_shape(&'a self) -> Self::ShapeType {
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
