use crate::prelude::*;

pub struct ParamTensor<F> {
    pub(crate) array: Array<F>,
    pub(crate) grad: Option<Array<F>>,
}

pub struct ParamsStorage<F> {
    pub(crate) storage: Vec<ParamTensor<F>>,
}

impl<F> ParamsStorage<F> {
    pub fn new(capacity: usize) -> ParamsStorage<F> {
        Self {
            storage: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, param: ParamTensor<F>) {
        self.storage.push(param);
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
