use std::marker::PhantomData;

use crate::prelude::*;

// Take Replace
pub enum TakeType<F> {
    Array(Array<F>),
    Metadata(Array<F>, TensorMetadata),
}

impl<F> TakeType<F> {
    pub fn to_array_ref<T>(&self) -> ArrayRef<'_, F, T> {
        match self {
            Self::Array(array) => ArrayRef {
                data: &array.data,
                offset: array.offset,
                shape: &array.shape,
                stride: &array.stride,
                _array_type: PhantomData::default(),
            },
            TakeType::Metadata(array, metadata) => ArrayRef {
                data: &array.data,
                offset: metadata.offset,
                shape: &metadata.shape,
                stride: &metadata.stride,
                _array_type: PhantomData::default(),
            },
        }
    }
}
