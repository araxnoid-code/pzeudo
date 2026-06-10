use crate::PzeudoDataTypeTrait;

pub trait ShapeTrait {
    type ShapeType;
    fn new(shape: Self::ShapeType) -> Self;
}

pub trait Arr<'s> {
    type ArrType;
    type ScalarType;
    type ShapeType: ShapeTrait;

    // desc
    fn get_array(&'s self) -> &'s Self::ArrType;
    fn get_shape(&'s self) -> Self::ShapeType;

    // initial
    fn zeros(shape: Self::ShapeType) -> Self;
    fn ones(shape: Self::ShapeType) -> Self;
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
