use std::marker::PhantomData;

use crate::{Arr, NDArrayDataType, NDArrayShape, PzeudoDataType, ShapeTrait};

pub struct NDArrayArr<'a, T>
where
    T: NDArrayDataType<'a>,
{
    inner: T,
    _phantom: PhantomData<&'a i32>,
}

impl<'a, T> NDArrayArr<'a, T>
where
    T: NDArrayDataType<'a>,
{
    pub fn new(inner: T) -> NDArrayArr<'a, T> {
        Self {
            inner,
            _phantom: PhantomData::default(),
        }
    }
}

impl<'a, T> Arr<'a> for NDArrayArr<'a, T>
where
    T: NDArrayDataType<'a, ScalarType = PzeudoDataType>,
{
    type ArrType = T;
    type ScalarType = T::ScalarType;
    type ShapeType = &'a [usize];

    // desc
    fn get_array(&'a self) -> &'a Self::ArrType {
        self.inner.get_array()
    }

    fn get_shape(&'a self) -> Self::ShapeType {
        self.inner.get_shape()
        // NDArrayShape {
        //     shape: self.inner.get_shape(),
        // }
    }

    // intial
    fn from_scalar(scalar: impl crate::PzeudoDataTypeTrait) -> Self {
        Self {
            inner: T::from_scalar(scalar.into_pzeudo_data_type()),
            _phantom: PhantomData::default(),
        }
    }

    // fn zeros(shape: Self::ShapeType) -> Self {
    //     Self {
    //         inner: T::zeros(shape.shape),
    //     }
    // }

    // fn ones(shape: Self::ShapeType) -> Self {
    //     Self {
    //         inner: T::ones(shape.shape),
    //     }
    // }

    // setter
    fn add_to(&mut self, rhs: &Self) {
        self.inner.add_to(&rhs.inner);
    }

    // element-wise ops
    fn add(&self, rhs: &Self) -> Self {
        let inner = self.inner.add(&rhs.inner);
        Self {
            inner,
            _phantom: PhantomData::default(),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        let inner = self.inner.sub(&rhs.inner);
        Self {
            inner,
            _phantom: PhantomData::default(),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        let inner = self.inner.div(&rhs.inner);
        Self {
            inner,
            _phantom: PhantomData::default(),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        let inner = self.inner.mul(&rhs.inner);
        Self {
            inner,
            _phantom: PhantomData::default(),
        }
    }

    // scalar element-wise ops
}
