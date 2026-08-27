use crate::prelude::*;
use num_traits::Zero;

#[derive(Clone, Copy)]
pub struct ReqGrad;

#[derive(Clone, Copy)]
pub struct ReqNoGrad;

#[derive(Clone, Copy)]
pub struct TrainPhase;

#[derive(Clone, Copy)]
pub struct EvalPhase;

pub trait ReqGradTrait<F>: Clone + Copy {
    fn zeros_grad(shape: &[usize]) -> Option<Array<F>>;

    fn is_grad(self) -> bool;

    fn into_zeros_grad_storage(
        &self,
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr>;

    fn zeros_grad_storage(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr>;
}

impl<F> ReqGradTrait<F> for ReqGrad
where
    F: Clone + Zero,
{
    fn zeros_grad(shape: &[usize]) -> Option<Array<F>> {
        Some(Array::zeros(shape))
    }

    fn is_grad(self) -> bool {
        true
    }

    fn into_zeros_grad_storage(
        &self,
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }

    fn zeros_grad_storage(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }
}

impl<F> ReqGradTrait<F> for TrainPhase
where
    F: Clone + Zero,
{
    fn zeros_grad(shape: &[usize]) -> Option<Array<F>> {
        Some(Array::zeros(shape))
    }

    fn is_grad(self) -> bool {
        true
    }

    fn into_zeros_grad_storage(
        &self,
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }

    fn zeros_grad_storage(
        shape: &[usize],
        storage: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        let grad = Array::zeros(shape);
        let grad_idx = Some(storage.push(ElementType::Grad(grad))?);
        Ok(grad_idx)
    }
}

impl<F> ReqGradTrait<F> for ReqNoGrad
where
    F: Clone + Zero,
{
    fn zeros_grad(_: &[usize]) -> Option<Array<F>> {
        None
    }

    fn is_grad(self) -> bool {
        false
    }

    fn into_zeros_grad_storage(
        &self,
        _: &[usize],
        _: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }

    fn zeros_grad_storage(
        _: &[usize],
        _: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }
}

impl<F> ReqGradTrait<F> for EvalPhase
where
    F: Clone + Zero,
{
    fn zeros_grad(_: &[usize]) -> Option<Array<F>> {
        None
    }

    fn is_grad(self) -> bool {
        false
    }

    fn into_zeros_grad_storage(
        &self,
        _: &[usize],
        _: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }

    fn zeros_grad_storage(
        _: &[usize],
        _: &mut ArrayStorage<F>,
    ) -> Result<Option<StorageType>, PzeudoErr> {
        Ok(None)
    }
}

impl<F> Tensor<F, Contiguous, ReqGrad> {
    pub fn no_grad(self) -> Result<Tensor<F, Contiguous, ReqNoGrad>, PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        let storage_type = self.grad_idx.ok_or(PzeudoErr::ReqGradErr(format!(
            "Tensor::no_grad. gradient tensor of type None"
        )))?;

        match storage_type {
            StorageType::View(_) => Err(PzeudoErr::ReqGradErr(format!(
                "Tensor::no_grad. cannot do Tensor::no_grad on tensor view"
            ))),
            StorageType::Param(idx) => storage.params_storage.no_grad(idx),
            StorageType::Arr(idx, grad_time) => storage.grad_storage.no_grad(
                idx,
                grad_time.ok_or(PzeudoErr::ReqGradErr(format!(
                    "Tensor::no_grad. tensor does not have grad_time"
                )))?,
            ),
        }?;

        drop(storage);
        let tensor = Tensor::_new(
            self.array_idx,
            None,
            self.shape,
            self.record_status,
            self.record,
            self.storage,
        );

        Ok(tensor)
    }
}

impl<F> Tensor<F, Contiguous, ReqNoGrad>
where
    F: Clone + Zero,
{
    pub fn with_grad(self) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        self.grad_idx.map_or(Ok(()), |_| {
            Err(PzeudoErr::ReqGradErr(format!(
                "Tensor::with_grad. tensor has gradient"
            )))
        })?;

        let grad_idx = match self.array_idx {
            StorageType::View(_) => Err(PzeudoErr::ReqGradErr(format!(
                "Tensor::no_grad. cannot do Tensor::no_grad on tensor view"
            ))),
            StorageType::Param(idx) => {
                let zeros = Array::<F>::zeros(&self.shape);
                storage.get_params_storage_mut().with_grad(idx, zeros)?;
                Ok(Some(StorageType::Param(idx)))
            }
            StorageType::Arr(_, _) => ReqGrad.into_zeros_grad_storage(&self.shape, &mut storage),
        }?;

        drop(storage);
        let tensor = Tensor::_new(
            self.array_idx,
            grad_idx,
            self.shape,
            self.record_status,
            self.record,
            self.storage,
        );
        Ok(tensor)
    }
}

pub trait PhaseStatus<F>: ReqGradTrait<F> {
    fn is_eval(self) -> bool;
}
impl<F> PhaseStatus<F> for TrainPhase
where
    F: Clone + Zero,
{
    fn is_eval(self) -> bool {
        false
    }
}
impl<F> PhaseStatus<F> for EvalPhase
where
    F: Clone + Zero,
{
    fn is_eval(self) -> bool {
        true
    }
}
