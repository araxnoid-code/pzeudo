use std::marker::PhantomData;

use crate::prelude::*;

pub fn metadata_to_array_ref<'a, F, T>(
    array_data: (usize, TensorType<'a, T>),
    storage: &'a ArrayStorage<F>,
) -> Result<ArrayRef<'a, F, T>, PzeudoErr> {
    let array = storage.get_element(array_data.0)?;

    if let TensorType::Contiguous = array_data.1 {
        Ok(ArrayRef {
            data: &array.data,
            offset: array.offset,
            shape: &array.shape,
            stride: &array.stride,
            _array_type: PhantomData::default(),
        })
    } else if let TensorType::View(metadata) = array_data.1 {
        Ok(ArrayRef {
            data: &array.data,
            offset: metadata.offset,
            shape: &metadata.shape,
            stride: &metadata.stride,
            _array_type: PhantomData::default(),
        })
    } else {
        panic!()
    }
}
