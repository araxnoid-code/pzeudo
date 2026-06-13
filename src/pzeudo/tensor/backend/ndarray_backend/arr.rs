use std::marker::PhantomData;

use crate::{Arr, NDArrayDataType, PzeudoDataType};

pub struct NDArrayArr<T>
where
    T: NDArrayDataType,
{
    inner: T,
}

impl<T> NDArrayArr<T>
where
    T: NDArrayDataType,
{
    pub fn new(inner: T) -> NDArrayArr<T> {
        Self { inner }
    }
}

impl<T> Arr for NDArrayArr<T>
where
    T: NDArrayDataType<ScalarType = PzeudoDataType>,
{
    type InnerArrType = T::ArrType;
    type ArrType = T;
    type ScalarType = T::ScalarType;
    type ShapeType = Vec<usize>;

    // desc
    fn get_array(&self) -> &Self::InnerArrType {
        self.inner.get_array()
    }

    fn get_shape(&self) -> Self::ShapeType {
        self.inner.get_shape().to_vec()
    }

    // intial
    fn from_scalar(scalar: impl crate::PzeudoDataTypeTrait) -> Self {
        Self {
            inner: T::from_scalar(scalar.into_pzeudo_data_type()),
        }
    }

    fn zeros(shape: Self::ShapeType) -> Self {
        Self {
            inner: T::zeros(shape),
        }
    }

    // fn ones(shape: Self::ShapeType) -> Self {
    //     Self {
    //         inner: T::ones(shape.shape),
    //     }
    // }

    // getter
    fn get_inner(&self) -> Self::InnerArrType {
        panic!()
    }

    // setter
    fn add_to(&mut self, rhs: &Self) {
        self.inner.add_to(&rhs.inner);
    }

    // element-wise ops
    fn add(&self, rhs: &Self) -> Self {
        let inner = self.inner.add(&rhs.inner);
        Self { inner }
    }

    fn sub(&self, rhs: &Self) -> Self {
        let inner = self.inner.sub(&rhs.inner);
        Self { inner }
    }

    fn div(&self, rhs: &Self) -> Self {
        let inner = self.inner.div(&rhs.inner);
        Self { inner }
    }

    fn mul(&self, rhs: &Self) -> Self {
        let inner = self.inner.mul(&rhs.inner);
        Self { inner }
    }
}
