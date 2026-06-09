use crate::{Arr, NDArrayDataType};

pub struct NDArrayArr<T>
where
    T: NDArrayDataType,
{
    inner: T,
}

impl<T> Arr for NDArrayArr<T>
where
    T: NDArrayDataType,
{
    fn add(&self, rhs: &Self) -> Self {
        let inner = self.inner.add(&rhs.inner);
        Self { inner }
    }
}
