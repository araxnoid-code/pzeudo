use std::marker::PhantomData;

use crate::prelude::*;

pub fn metadata_to_array_ref<'a, F, T>(
    tensor_metadata: &'a TensorMetadata,
    idx: usize,
    storage: &'a ArrayStorage<F>,
) -> Result<ArrayRef<'a, F, T>, PzeudoErr> {
    let data = storage.get_element(idx)?;

    Ok(ArrayRef {
        data,
        offset: tensor_metadata.offset,
        shape: &tensor_metadata.shape,
        stride: &tensor_metadata.stride,
        _array_type: PhantomData::default(),
    })
}
