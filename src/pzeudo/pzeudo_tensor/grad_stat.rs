use crate::prelude::*;
use num_traits::Zero;

pub struct Grad;
pub struct NoGrad;

pub trait GradStatTrait<F> {
    fn zeros_grad(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr>;
}

impl<F> GradStatTrait<F> for Grad
where
    F: Clone + Zero,
{
    fn zeros_grad(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }
}

impl<F> GradStatTrait<F> for NoGrad
where
    F: Clone + Zero,
{
    fn zeros_grad(_: &[usize], _: &mut ArrayStorage<F>) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }
}
