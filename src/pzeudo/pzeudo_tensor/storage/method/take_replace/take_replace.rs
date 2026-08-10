use crate::{prelude::*, pzeudo_tensor::storage::method::take_replace::TakeType};

impl<F> ArrayStorage<F> {
    pub fn take_grad(&mut self, storage_type: StorageType) -> Result<TakeType<F>, PzeudoErr> {
        match storage_type {
            StorageType::Arr(idx, _) => {
                Ok(TakeType::Array(self.get_grad_storage_mut().take_grad(idx)?))
            }
            StorageType::Param(idx) => Ok(TakeType::Array(
                self.get_params_storage_mut().take_grad(idx)?,
            )),
            StorageType::View(idx) => {
                let view = self.view_storage.take_metadata(idx)?;

                match view.arr_index {
                    ViewStorageType::Param(idx) => {
                        let params = self.get_params_storage_mut().take_grad(idx)?;
                        Ok(TakeType::Metadata(params, view))
                    }
                    ViewStorageType::Storage(idx, _) => {
                        let grad = self.get_grad_storage_mut().take_grad(idx)?;
                        Ok(TakeType::Metadata(grad, view))
                    }
                }
                //
            }
        }
    }

    pub fn replace_grad(
        &mut self,
        storage_type: StorageType,
        take_grad: TakeType<F>,
    ) -> Result<(), PzeudoErr> {
        match storage_type {
            StorageType::Arr(idx, _) => match take_grad {
                TakeType::Array(grad) => {
                    self.get_grad_storage_mut().replace_grad(idx, grad)?;
                }
                TakeType::Metadata(_, _) => {
                    return Err(PzeudoErr::StorageErr(format!(
                        "ArrayStorage::replace_grad. TakeType::Metadata does not match StorageType::Arr"
                    )));
                }
            },
            StorageType::Param(idx) => match take_grad {
                TakeType::Array(grad) => self.get_params_storage_mut().replace_grad(idx, grad)?,
                TakeType::Metadata(_, _) => {
                    return Err(PzeudoErr::StorageErr(format!(
                        "ArrayStorage::replace_grad. TakeType::Metadata does not match StorageType::Param"
                    )));
                }
            },
            StorageType::View(v_idx) => match take_grad {
                TakeType::Metadata(grad, metadata) => {
                    match metadata.arr_index {
                        ViewStorageType::Param(idx) => {
                            self.get_params_storage_mut().replace_grad(idx, grad)?;
                        }
                        ViewStorageType::Storage(idx, _) => {
                            self.get_grad_storage_mut().replace_grad(idx, grad)?;
                        }
                    }

                    self.view_storage.replace_metadata(v_idx, metadata)?;
                }
                TakeType::Array(_) => {
                    return Err(PzeudoErr::StorageErr(format!(
                        "ArrayStorage::replace_grad. TakeType::Array does not match StorageType::View"
                    )));
                }
            },
        };

        Ok(())
    }
}
