use crate::{PzeudoBackend, Tensor, tensor::ops::BackwardLabel};

impl<B> Tensor<B>
where
    B: PzeudoBackend,
{
    pub fn mul(&self, rhs: &Self) -> Tensor<B> {
        let lhs = self.inner.clone();
        let rhs = rhs.inner.clone();
        let output = lhs.read().unwrap().mul(rhs.read().as_ref().unwrap(), true);
        Tensor::new(output, Some(BackwardLabel::Mul(lhs, rhs)))
    }
}
