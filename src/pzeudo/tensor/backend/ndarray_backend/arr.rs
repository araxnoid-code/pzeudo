use crate::{Arr, NDArrayDataType, ShapeTrait};

pub struct Shape<'a> {
    shape: &'a [usize],
}

impl<'a> ShapeTrait for Shape<'a> {
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
    T: NDArrayDataType,
{
    type ShapeType = Shape<'s>;
    // desc
    fn get_shape(&'s self) -> Self::ShapeType {
        Shape {
            shape: self.inner.get_shape(),
        }
    }
    // fn get_shape<'a>(&'a self) -> Self::ShapeType<'a> {
    // Shape {
    //     shape: self.inner.get_shape(),
    // }
    // }

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
