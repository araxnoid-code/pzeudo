use crate::{StorageType, ViewStorageType};

#[derive(Clone)]
pub struct TensorMetadata {
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) arr_index: ViewStorageType,
}

impl TensorMetadata {
    pub fn new(
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
        arr_index: ViewStorageType,
    ) -> TensorMetadata {
        Self {
            offset,
            shape,
            stride,
            arr_index,
        }
    }
}
