use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s, ScalarType = PzeudoDataType>,
    B: PzeudoBackend<'s, A>,
{
    pub fn sub(&self, rhs: &Self) -> Tensor<'s, A, B> {
        // let lhs = self.backend;
        // let rhs = rhs.backend;
        let output = self.backend.sub(&rhs.backend);
        Tensor::new(output, None)
    }
}
