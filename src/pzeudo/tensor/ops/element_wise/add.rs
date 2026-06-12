use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<A, B> Tensor<A, B>
where
    A: Arr<ScalarType = PzeudoDataType>,
    B: PzeudoBackend<A>,
{
    pub fn add(&self, rhs: &Self) -> Tensor<A, B> {
        let output = self.backend.add(&rhs.backend);
        Tensor::new(output, None)
    }
}
