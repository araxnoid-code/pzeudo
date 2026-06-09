use crate::{Arr, PzeudoBackend, Tensor, tensor::ops::BackwardLabel};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s>,
    B: PzeudoBackend<'s, A>,
{
    pub fn sub(&self, rhs: &Self) -> Tensor<'s, A, B> {
        let lhs = self.inner.clone();
        let rhs = rhs.inner.clone();
        let output = lhs.read().unwrap().sub(rhs.read().as_ref().unwrap());
        Tensor::new(output, Some(BackwardLabel::Sub(lhs, rhs)))
    }
}
