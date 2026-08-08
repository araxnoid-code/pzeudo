use crate::prelude::*;

pub struct ViewStorage {
    storage: Vec<Option<TensorMetadata>>,
    empty_idx: Vec<usize>,
}

impl ViewStorage {
    pub(crate) fn new(capacity: Option<usize>) -> ViewStorage {
        Self {
            storage: Vec::with_capacity(capacity.unwrap_or(0)),
            empty_idx: Vec::new(),
        }
    }

    pub(crate) fn take_metadata(&mut self, idx: usize) -> Result<TensorMetadata, PzeudoErr> {
        Ok(self.storage
            .get_mut(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::take_grad. index {idx} points to an invalid location on gradient storage."
            )))?
            .take()
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::take_grad. index {idx} points to elements that have the value None in gradient storage."
            )))?)
    }

    pub(crate) fn replace_metadata(
        &mut self,
        idx: usize,
        metadata: TensorMetadata,
    ) -> Result<(), PzeudoErr> {
        let space = self.storage.get_mut(idx).ok_or(PzeudoErr::StorageErr(format!(
            "GradStorage::replace_grad. index {idx} points to an invalid location on gradient storage."
        )))?;

        if space.is_some() {
            return Err(PzeudoErr::StorageErr(format!(
                "GradStorage::replace_grad. Index {idx} points to the location where the gradient is stored in gradient storage."
            )));
        }

        space.replace(metadata);
        Ok(())
    }

    pub(crate) fn push_metadata(&mut self, metadata: TensorMetadata) -> Result<usize, PzeudoErr> {
        if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoErr::StorageErr(format!(
                    "ArrayStorage::push. The problem occurs because the index {idx} obtained from empty_idx points to an element that still has a value."
                )));
            }

            return Ok(idx);
        } else {
            self.storage.push(Some(metadata));
            return Ok(self.storage.len() - 1);
        };
    }

    pub(crate) fn get_metadata(&self, idx: usize) -> Result<&TensorMetadata, PzeudoErr> {
        let array = self
            .storage
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to an invalid location on gradient storage."
            )))?
            .as_ref()
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to elements that have the value None in gradient storage."
            )))?;

        Ok(array)
    }

    pub fn clear(&mut self) {
        self.storage.clear();
        self.empty_idx.clear();
    }
}

// impl<F> Deref for GradStorage<F> {
//     type Target = Vec<Option<Array<F>>>;
//     fn deref(&self) -> &Self::Target {
//         &self.storage
//     }
// }

// impl<F> DerefMut for GradStorage<F> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.storage
//     }
// }
