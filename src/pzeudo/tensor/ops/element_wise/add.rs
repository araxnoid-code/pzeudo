use crate::{Arr, PzeudoBackend, Tensor, tensor::ops::BackwardLabel};

impl<A, B> Tensor<A, B>
where
    A: Arr,
    B: PzeudoBackend<A>,
{
    pub fn add(&self, rhs: &Self) -> Tensor<A, B> {
        let lhs = self.inner.clone();
        let rhs = rhs.inner.clone();
        let output = lhs.read().unwrap().add(rhs.read().as_ref().unwrap());
        Tensor::new(output, Some(BackwardLabel::Add(lhs, rhs)))
    }
}
