use std::marker::PhantomData;

use crate::{ArrayRef, ArrayStorage, TensorMetadata};

pub fn metadata_to_array_ref<'a, F, T>(
    tensor_metadata: &'a TensorMetadata,
    idx: usize,
    storage: &'a ArrayStorage<F>,
) -> ArrayRef<'a, F, T> {
    let storage = storage.get_storage().get(idx).unwrap().as_ref().unwrap();

    ArrayRef {
        data: storage,
        offset: tensor_metadata.offset,
        shape: &tensor_metadata.shape,
        stride: &tensor_metadata.stride,
        _array_type: PhantomData::default(),
    }
}
