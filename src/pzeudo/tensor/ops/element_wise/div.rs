use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s, ScalarType = PzeudoDataType>,
    B: PzeudoBackend<'s, A>,
{
    pub fn div(&self, rhs: &Self) -> Tensor<'s, A, B> {
        // let lhs = self.backend.clone();
        // let rhs = rhs.backend.clone();
        let output = self.backend.div(&rhs.backend);
        Tensor::new(output, None)
    }
}
