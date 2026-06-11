use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor, tensor::ops::BackwardLabel};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s, ScalarType = PzeudoDataType>,
    B: PzeudoBackend<'s, A>,
{
    pub fn div(&self, rhs: &Self) -> Tensor<'s, A, B> {
        let lhs = self.backend.clone();
        let rhs = rhs.backend.clone();
        let output = lhs.read().unwrap().div(rhs.read().as_ref().unwrap());
        Tensor::new(output, Some(BackwardLabel::Div(lhs, rhs)))
    }
}
