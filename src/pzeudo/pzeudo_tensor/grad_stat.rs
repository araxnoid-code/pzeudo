use crate::prelude::*;
use num_traits::Zero;

pub struct Grad;
pub struct NoGrad;

pub trait ReqGradTrait<F> {
    fn into_zeros_grad(
        self,
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr>;

    fn zeros_grad(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr>;
}

impl<F> ReqGradTrait<F> for Grad
where
    F: Clone + Zero,
{
    fn into_zeros_grad(
        self,
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }

    fn zeros_grad(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }
}

impl<F> ReqGradTrait<F> for NoGrad
where
    F: Clone + Zero,
{
    fn into_zeros_grad(
        self,
        _: &[usize],
        _: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }

    fn zeros_grad(_: &[usize], _: &mut ArrayStorage<F>) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }
}

impl<F> Tensor<F, Contiguous, Grad> {
    pub fn no_grad(&mut self) -> Result<(), PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        let storage_type = self.grad_idx.ok_or(PzeudoErr::NoGradErr(format!(
            "Tensor::no_grad. gradient tensor of type None"
        )))?;

        match storage_type {
            StorageType::Permanent(_) => Err(PzeudoErr::NoGradErr(format!(
                "Tensor::no_grad. can't do Tensor::no_grad on permanent tensor"
            ))),
            StorageType::View(_) => Err(PzeudoErr::NoGradErr(format!(
                "Tensor::no_grad. cannot do Tensor::no_grad on tensor view"
            ))),
            StorageType::Arr(idx, grad_time) => storage.grad_storage.no_grad(
                idx,
                grad_time.ok_or(PzeudoErr::NoGradErr(format!(
                    "Tensor::no_grad. tensor does not have grad_time"
                )))?,
            ),
        }?;

        self.grad_idx = None;
        Ok(())
    }
}

impl<F> Tensor<F, Contiguous, NoGrad>
where
    F: Clone + Zero,
{
    pub fn with_grad(&mut self) -> Result<(), PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        self.grad_idx.map_or(Ok(()), |_| {
            Err(PzeudoErr::WithGradErr(format!(
                "Tensor::with_grad. tensor has gradient"
            )))
        })?;

        self.grad_idx = Grad.into_zeros_grad(&self.shape, &mut storage)?;
        Ok(())
    }
}
