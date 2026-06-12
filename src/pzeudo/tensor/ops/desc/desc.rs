use std::sync::RwLockReadGuard;

use crate::{Arr, PzeudoBackend, Tensor};

impl<A, B> Tensor<A, B>
where
    A: Arr,
    B: PzeudoBackend<A>,
{
    pub fn get_array(&self) -> &A::InnerArrType {
        self.backend.get_array()
    }

    pub fn get_grad(&self) -> Option<&A::InnerArrType> {
        self.backend.get_grad()
    }

    pub fn get_shape(&self) -> B::ShapeType {
        self.backend.get_shape()
    }
}
