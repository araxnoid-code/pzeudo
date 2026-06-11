use std::sync::RwLockReadGuard;

use crate::{Arr, PzeudoBackend, ShapeTrait, Tensor};

impl<'a, A, B> Tensor<'a, A, B>
where
    A: Arr<'a>,
    B: PzeudoBackend<'a, A>,
{
    pub fn get_array(&'a self) -> &'a A::InnerArrType {
        self.backend.get_array()
    }

    pub fn get_grad(&'a self) -> Option<&'a A::InnerArrType> {
        self.backend.get_grad()
    }

    pub fn get_shape(&'a self) -> B::ShapeType {
        self.backend.get_shape()
    }
}
