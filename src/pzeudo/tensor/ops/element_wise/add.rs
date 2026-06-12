use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s, ScalarType = PzeudoDataType>,
    B: PzeudoBackend<'s, A>,
{
    pub fn add(&self, rhs: &Self) -> Tensor<'s, A, B> {
        let output = self.backend.add(&rhs.backend);
        Tensor::new(output, None)
    }
}
