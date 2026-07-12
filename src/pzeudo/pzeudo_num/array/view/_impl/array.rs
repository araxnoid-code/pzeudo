use crate::{ArrayTrait, ArrayView, Metadata};

impl<F> ArrayTrait<F> for ArrayView<'_, F> {
    fn get_metadata(&self) -> crate::Metadata<'_, F> {
        Metadata {
            data: self.data,
            offset: self.offset,
            shape: &self.shape,
            stride: &self.stride,
        }
    }
}
