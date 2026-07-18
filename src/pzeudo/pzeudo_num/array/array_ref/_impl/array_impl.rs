use crate::prelude::*;

impl<F> ArrayTrait<F> for ArrayRef<'_, F, View> {
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

impl<F> ArrayTrait<F> for ArrayRef<'_, F, Contiguous> {
    fn get_metadata(&self) -> Metadata<'_, F> {
        let metadata = Metadata {
            data: self.data,
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        };
        metadata
    }

    fn linear_index(&self, index: usize) -> Result<F, PzeudoErr>
    where
        F: Copy,
    {
        Ok(self.data[index])
    }
}
