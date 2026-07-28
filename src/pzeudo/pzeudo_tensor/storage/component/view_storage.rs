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

    pub(crate) fn push_metadata(&mut self, metadata: TensorMetadata) -> Result<usize, PzeudoErr> {
        if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoErr::StoragePushErr(format!(
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
            .ok_or(PzeudoErr::GradStorageGetErr(format!(
                "GradStorage::get_grad. index {idx} points to an invalid location on gradient storage."
            )))?
            .as_ref()
            .ok_or(PzeudoErr::GradStorageGetErr(format!(
                "GradStorage::get_grad. index {idx} points to elements that have the value None in gradient storage."
            )))?;

        Ok(array)
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
