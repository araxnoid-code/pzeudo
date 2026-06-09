use crate::{Arr, NDArrayDataType};

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
    T: NDArrayDataType,
{
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
