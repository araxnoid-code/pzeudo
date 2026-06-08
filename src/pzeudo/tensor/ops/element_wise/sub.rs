use crate::{PzeudoBackend, Tensor, tensor::ops::BackwardLabel};

impl<B> Tensor<B>
where
    B: PzeudoBackend,
{
    pub fn sub(&self, rhs: &Self) -> Tensor<B> {
        let lhs = self.inner.clone();
        let rhs = rhs.inner.clone();
        let output = lhs.read().unwrap().sub(rhs.read().as_ref().unwrap());
        Tensor::new(output, Some(BackwardLabel::Sub(lhs, rhs)))
    }
}
