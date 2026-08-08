use crate::prelude::*;

impl<F> ArrayStorage<F> {
    pub fn take_grad(&mut self, storage_type: StorageType) -> Result<Array<F>, PzeudoErr> {
        match storage_type {
            StorageType::Arr(idx, _) => Ok(self.get_grad_storage_mut().take_grad(idx)?),
            StorageType::Param(idx) => Ok(self.get_params_storage_mut().take_grad(idx)?),
            StorageType::View(_) => Err(PzeudoErr::StorageErr(format!(
                "ArrayStorage::take_grad. Cannot retrieve the gradient from the view (the view does not have a gradient)."
            ))),
        }
    }

    pub fn replace_grad(
        &mut self,
        storage_type: StorageType,
        grad: Array<F>,
    ) -> Result<(), PzeudoErr> {
        match storage_type {
            StorageType::Arr(idx, _) => Ok(self.get_grad_storage_mut().replace_grad(idx, grad)?),
            StorageType::Param(idx) => Ok(self.get_params_storage_mut().replace_grad(idx, grad)?),
            StorageType::View(_) => Err(PzeudoErr::StorageErr(format!(
                "ArrayStorage::take_grad. Cannot retrieve the gradient from the view (the view does not have a gradient)."
            ))),
        }
    }
}
