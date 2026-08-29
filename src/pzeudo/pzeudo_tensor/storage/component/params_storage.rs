use crate::prelude::*;

pub struct ParamTensor<F> {
    pub(crate) array: Array<F>,
    pub(crate) grad: Option<Array<F>>,
}

pub struct ParamsStorage<F> {
    pub(crate) update: Vec<bool>,
    pub(crate) storage: Vec<ParamTensor<F>>,
}

impl<F> ParamsStorage<F> {
    pub fn new(capacity: usize) -> ParamsStorage<F> {
        Self {
            update: Vec::with_capacity(capacity),
            storage: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn take_grad(&mut self, idx: usize) -> Result<Array<F>, PzeudoErr> {
        Ok(self.storage
            .get_mut(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "ParamsStorage::take_params_grad. index {idx} points to an invalid location."
            )))?.grad
        .take()
        .ok_or(PzeudoErr::StorageErr(format!(
            "ParamsStorage::take_params_grad. index {idx} points to elements that have the value None."
        )))?)
    }

    pub(crate) fn replace_grad(&mut self, idx: usize, grad: Array<F>) -> Result<(), PzeudoErr> {
        let space = self.storage.get_mut(idx).ok_or(PzeudoErr::StorageErr(format!(
            "ParamsStorage::replace_params_grad. index {idx} points to an invalid location on gradient storage."
        )))?;

        if space.grad.is_some() {
            return Err(PzeudoErr::StorageErr(format!(
                "ParamsStorage::replace_params_grad. Index {idx} points to the location where the gradient is stored in gradient storage."
            )));
        }

        space.grad.replace(grad);
        Ok(())
    }

    pub fn push(&mut self, param: ParamTensor<F>) {
        self.storage.push(param);
        self.update.push(false);
    }

    pub fn set_update(&mut self, idx: usize, status: bool) -> Result<(), PzeudoErr> {
        *self.update
            .get_mut(idx)
            .ok_or(PzeudoErr::StorageErr(format!("ParamsStorage::set_update. index {idx} points to an invalid location on gradient storage(status).")))? = status;
        Ok(())
    }

    pub fn is_update(&self, idx: usize) -> Result<bool, PzeudoErr> {
        Ok(*self.update.get(idx).ok_or(PzeudoErr::StorageErr(format!(
            "ParamsStorage::is_update. Index {idx} points to an invalid location in params storage(update)."
        )))?)
    }

    pub fn no_grad(&mut self, idx: usize) -> Result<(), PzeudoErr> {
        self.storage
            .get_mut(idx)
            .ok_or(PzeudoErr::StorageErr(format!("ParamsStorage::no_grad. Index {idx} points to an invalid location in params storage.")))?
            .grad
            .take()
            .ok_or(PzeudoErr::StorageErr(format!("ParamsStorage::no_grad. Index {idx} refers to a tensor parameter with NoGrad status.")))?;

        Ok(())
    }

    pub fn with_grad(&mut self, idx: usize, grad: Array<F>) -> Result<(), PzeudoErr> {
        *self.storage
            .get_mut(idx)
            .ok_or(PzeudoErr::StorageErr(format!("ParamsStorage::no_grad. Index {idx} points to an invalid location in params storage.")))?
            .grad
            .as_mut()
            .ok_or(PzeudoErr::StorageErr(format!("ParamsStorage::no_grad. Index {idx} refers to a tensor parameter with NoGrad status.")))?
        = grad;

        Ok(())
    }
}
