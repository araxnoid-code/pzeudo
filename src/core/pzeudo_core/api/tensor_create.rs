use crate::{PzeudoCore, Tensor};

impl<const N: usize, const MAX_RING_BUFFER: usize> PzeudoCore<N, MAX_RING_BUFFER> {
    pub fn create_tensor_from_value(&self, value: f64) -> Tensor {
        self.tensor_core.create_tensor_from_value(value)
    }
}
