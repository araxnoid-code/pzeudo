use crate::{Arr, NDArrayDataType, PzeudoDataType, ShapeTrait};

pub struct NDArrayShape<'a> {
    pub shape: &'a [usize],
}

impl<'a> ShapeTrait for NDArrayShape<'a> {
    type ShapeType = &'a [usize];
    fn new(shape: Self::ShapeType) -> Self {
        Self { shape }
    }
}

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

impl<'s, T> Arr<'s> for NDArrayArr<T>
where
    T: NDArrayDataType<ScalarType = PzeudoDataType>,
{
    type ArrType = T;
    type ScalarType = T::ScalarType;
    type ShapeType = NDArrayShape<'s>;
    // desc
    fn get_array(&'s self) -> &'s Self::ArrType {
        self.inner.get_array()
    }

    fn get_shape(&'s self) -> Self::ShapeType {
        NDArrayShape {
            shape: self.inner.get_shape(),
        }
    }

    // intial
    fn from_scalar(scalar: impl crate::PzeudoDataTypeTrait) -> Self {
        Self {
            inner: T::from_scalar(scalar.into_pzeudo_data_type()),
        }
    }

    fn zeros(shape: Self::ShapeType) -> Self {
        Self {
            inner: T::zeros(shape.shape),
        }
    }

    fn ones(shape: Self::ShapeType) -> Self {
        Self {
            inner: T::ones(shape.shape),
        }
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

    // scalar element-wise ops
    // fn scalar_mul(&self, rhs: Self::ScalarType) -> Self {
    //     Self {
    //         inner: self.inner.scalar_mul(rhs),
    //     }
    // }
}
