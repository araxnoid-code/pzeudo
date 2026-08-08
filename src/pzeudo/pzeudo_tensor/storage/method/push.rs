use crate::prelude::*;

impl<F> ArrayStorage<F> {
    pub fn push_param_tensor(&mut self, array: Array<F>, grad: Option<Array<F>>) -> StorageType {
        let idx = self.params_storage.storage.len();
        self.params_storage.push(ParamTensor { array, grad });
        StorageType::Param(idx)
    }

    pub fn push(&mut self, element: ElementType<F>) -> Result<StorageType, PzeudoErr> {
        match element {
            ElementType::Grad(array) => {
                let (idx, grad_time) = self.grad_storage.push_grad(array)?;
                Ok(StorageType::Arr(idx, Some(grad_time)))
            }

            ElementType::Arr(array) => {
                let idx = self.arr_storage.push_arr(array)?;

                return Ok(StorageType::Arr(idx, None));
            }
            ElementType::View(metadata) => {
                let idx = self.view_storage.push_metadata(metadata)?;
                return Ok(StorageType::View(idx));
            }
        }
    }
}
