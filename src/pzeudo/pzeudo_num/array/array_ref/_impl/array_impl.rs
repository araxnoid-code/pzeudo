use crate::prelude::*;

impl<F, T> ArrayTrait<F> for ArrayRef<'_, F, T> {
    fn get_metadata(&self) -> Metadata<'_, F> {
        let metadata = Metadata {
            data: self.data,
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        };
        metadata
    }
}
