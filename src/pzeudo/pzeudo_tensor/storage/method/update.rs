use crate::prelude::*;

impl<F> ArrayStorage<F> {
    pub fn set_grad_update(
        &mut self,
        element_type: StorageType,
        status: bool,
    ) -> Result<(), PzeudoErr> {
        match element_type {
            StorageType::Arr(idx, _) => {
                self.grad_storage.set_update(idx, status)?;
            }
            StorageType::View(m_idx) => {
                let metadata = self.view_storage.get_metadata(m_idx)?;
                match metadata.arr_index {
                    ViewStorageType::Storage(idx, _) => {
                        self.grad_storage.set_update(idx, status)?;
                    }
                    ViewStorageType::Param(idx) => {
                        self.params_storage.set_update(idx, status)?;
                    }
                }
            }
            StorageType::Param(idx) => {
                self.params_storage.set_update(idx, status)?;
            }
        }

        Ok(())
    }
}
