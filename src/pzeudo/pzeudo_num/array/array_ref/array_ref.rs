use std::marker::PhantomData;

use crate::Array;

pub struct ArrayRef<'a, F, T> {
    pub(crate) data: &'a [F],
    pub(crate) offset: usize,
    pub(crate) stride: &'a [usize],
    pub(crate) shape: &'a [usize],
    pub(crate) _array_type: PhantomData<T>,
}

impl<'a, F, T> ArrayRef<'a, F, T> {
    pub fn into_array(self) -> Array<F>
    where
        F: Clone,
    {
        Array {
            data: self.data.to_vec(),
            offset: self.offset,
            shape: self.shape.to_vec(),
            stride: self.stride.to_vec(),
        }
    }
}
