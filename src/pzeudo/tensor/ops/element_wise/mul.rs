use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s, ScalarType = PzeudoDataType>,
    B: PzeudoBackend<'s, A>,
{
    pub fn mul(&self, rhs: &Self) -> Tensor<'s, A, B> {
        let output = self.backend.mul(&rhs.backend);
        Tensor::new(output, None)
    }
}
