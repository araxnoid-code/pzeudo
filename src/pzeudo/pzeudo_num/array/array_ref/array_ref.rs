use std::marker::PhantomData;

use crate::prelude::*;

pub struct ArrayRef<'a, F, T> {
    pub(crate) data: &'a [F],
    pub(crate) offset: usize,
    pub(crate) stride: &'a [usize],
    pub(crate) shape: &'a [usize],
    pub(crate) _array_type: PhantomData<T>,
}

impl<'a, F> ArrayRef<'a, F, Contiguous> {
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

impl<'a, F> ArrayRef<'a, F, View> {
    pub fn into_array(self) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy,
    {
        let len = self.shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(self.linear_index(i)?);
        }

        Ok(Array {
            data,
            offset: self.offset,
            shape: self.shape.to_vec(),
            stride: self.stride.to_vec(),
        })
    }
}
