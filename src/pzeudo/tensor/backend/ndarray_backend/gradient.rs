use crate::NDArrayDataType;

pub struct NDArrayGradient<T>
where
    T: NDArrayDataType,
{
    inner: T,
}

// impl<T> Gradient for NDArrayGradient<T> where T: NDArrayDataType {}
