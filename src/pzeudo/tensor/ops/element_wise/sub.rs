use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<A, B> Tensor<A, B>
where
    A: Arr<ScalarType = PzeudoDataType>,
    B: PzeudoBackend<A>,
{
    pub fn sub(&self, rhs: &Self) -> Tensor<A, B> {
        // let lhs = self.backend;
        // let rhs = rhs.backend;
        let output = self.backend.sub(&rhs.backend);
        Tensor::new(output, None)
    }
}
