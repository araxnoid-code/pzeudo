use crate::{PzeudoDataTypeTrait, ShapeTrait};

pub trait Arr<'a> {
    type ArrType;
    type ScalarType;
    type ShapeType;

    // desc
    fn get_array(&'a self) -> &'a Self::ArrType;
    fn get_shape(&'a self) -> Self::ShapeType;

    // initial
    // fn zeros(shape: Self::ShapeType) -> Self;
    // fn ones(shape: Self::ShapeType) -> Self;
    fn from_scalar(scalar: impl PzeudoDataTypeTrait) -> Self;

    // setter
    fn add_to(&mut self, rhs: &Self);

    // element wise-ops
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;

    // scalar wise-ops
    // fn scalar_mul(&self, rhs: Self::ScalarType) -> Self;
}
