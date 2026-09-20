use crate::{ArrayTrait, ArrayView, PzeudoErr, get_broadcast_dim};

pub trait OpsBroadcast<F>: ArrayTrait<F> {
    fn broadcast(&self, to: &[usize]) -> Result<ArrayView<'_, F>, PzeudoErr> {
        let metadata = self.get_metadata();
        let broadcast_dim = get_broadcast_dim(metadata.shape, to)?;

        let new_shape = to.to_vec();
        let d = to.len() - metadata.shape.len();
        let mut new_stride = vec![0; d];
        new_stride.extend_from_slice(metadata.stride);
        broadcast_dim.iter().for_each(|dim| new_stride[*dim] = 0);

        let array = ArrayView {
            data: metadata.data,
            offset: metadata.offset,
            shape: new_shape,
            stride: new_stride,
        };

        Ok(array)
    }
}
