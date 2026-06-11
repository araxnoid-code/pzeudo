use std::fmt::Display;

pub trait NDArrayDataType<'a> {
    type ArrType: Display;
    type ScalarType;

    // desc
    fn get_array(&self) -> &Self::ArrType;
    fn get_shape(&'a self) -> &'a [usize];

    // initial
    fn zeros(shape: &'a [usize]) -> Self;
    fn ones(shape: &'a [usize]) -> Self;
    fn from_scalar(scalar: Self::ScalarType) -> Self;

    // setter
    fn add_to(&mut self, rhs: &Self);

    // element-wise ops
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;

    // scalar element-wise ops
    // fn add_mul<T>(&self, rhs: T) -> Self;
    // fn scalar_mul(&self, rhs: Self::ScalarType) -> Self;
}
