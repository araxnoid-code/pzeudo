use std::{fmt::Debug, marker::PhantomData};

use crate::{Arr, NDArrayArr, NDArrayDataType, PzeudoBackend, PzeudoDataType, ShapeTrait};

pub struct NDArrayBackend<'a, A>
where
    A: Arr<'a>,
{
    pub(crate) arr: A,
    pub(crate) grad: Option<A>,
    _phantom: PhantomData<&'a A>,
}

impl<'a, A> NDArrayBackend<'a, A>
where
    A: Arr<'a>,
{
    pub fn new(arr: A, grad: Option<A>) -> NDArrayBackend<'a, A> {
        Self {
            arr,
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

    // initial
    // fn zeros(shape: A::ShapeType) -> Self {
    //     panic!()
    //     // Self {
    //     //     arr: Self::BackendArrType::zeros(shape),
    //     // }
    // }

    fn arr_into(arr: A, grad: bool) -> Self {
        let grad = if grad {
            panic!()
            // Some(A::zeros(arr.get_shape()))
        } else {
            None
        };

        Self::new(arr, grad)
    }

    // desc
    fn get_shape(&'a self) -> Self::ShapeType {
        self.arr.get_shape()
    }

    // innner getter
    fn backend() -> impl Debug {
        "ndarray"
    }

    fn get_backend_arr(&self) -> &A {
        &self.arr
    }

    fn get_backend_grad_as_mut(&mut self) -> &mut Option<A> {
        &mut self.grad
    }
}
