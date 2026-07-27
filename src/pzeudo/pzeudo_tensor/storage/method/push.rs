use crate::prelude::*;

impl<F> ArrayStorage<F> {
    pub fn push_permanent_tensor(&mut self, array: Array<F>, grad: Array<F>) -> StorageType {
        let idx = self.permanent_storage.len();
        self.permanent_storage.push(PermanentTensor { array, grad });
        StorageType::Permanent(idx)
    }

    pub fn push(&mut self, element: ElementType<F>) -> Result<StorageType, PzeudoErr> {
        match element {
            ElementType::Grad(array) => {
                let idx = self.grad_storage.grad_push(array)?;
                Ok(StorageType::Arr(idx))
            }

            ElementType::Arr(array) => {
                let idx = self.arr_storage.push_arr(array)?;

                return Ok(StorageType::Arr(idx));
            }
            ElementType::View(metadata) => {
                let idx = self.view_storage.push_metadata(metadata)?;
                return Ok(StorageType::View(idx));
            }
        }
    }
}
