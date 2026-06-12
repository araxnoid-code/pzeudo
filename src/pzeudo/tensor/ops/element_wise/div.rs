use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<A, B> Tensor<A, B>
where
    A: Arr<ScalarType = PzeudoDataType>,
    B: PzeudoBackend<A>,
{
    pub fn div(&self, rhs: &Self) -> Tensor<A, B> {
        // let lhs = self.backend.clone();
        // let rhs = rhs.backend.clone();
        let output = self.backend.div(&rhs.backend);
        Tensor::new(output, None)
    }
}
