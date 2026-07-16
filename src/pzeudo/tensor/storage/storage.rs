use crate::{Array, PzeudoTensorErr};

pub struct ArrayStorage<F> {
    storage: Vec<Option<Array<F>>>,
    empty_idx: Vec<usize>,
}

impl<F> ArrayStorage<F> {
    pub fn new(capacity: Option<usize>) -> ArrayStorage<F> {
        let len = capacity.unwrap_or(1);
        Self {
            storage: Vec::with_capacity(len),
            empty_idx: Vec::new(),
        }
    }

    pub fn get_storage(&self) -> &Vec<Option<Array<F>>> {
        &self.storage
    }

    pub fn get_mut_storage(&mut self) -> &mut Vec<Option<Array<F>>> {
        &mut self.storage
    }

    pub fn get_empty_idx(&self) -> &Vec<usize> {
        &self.empty_idx
    }

    pub fn get_mut_empty_idx(&mut self) -> &mut Vec<usize> {
        &mut self.empty_idx
    }

    pub fn push(&mut self, element: Array<F>) -> Result<usize, PzeudoTensorErr> {
        let idx = if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoTensorErr::StoragePushErr(format!(
                    "ArrayStorage::push. The problem occurs because the index {idx} obtained from empty_idx points to an element that still has a value."
                )));
            }

            self.storage[idx].replace(element);
            idx
        } else {
            self.storage.push(Some(element));
            self.storage.len() - 1
        };

        Ok(idx)
    }

    pub fn remove(&mut self, idx: usize) -> Result<(), PzeudoTensorErr> {
        if self.storage[idx].is_none() {
            return Err(PzeudoTensorErr::StorageRemoveErr(format!(
                "ArrayStorage::remove. index {idx} points to an element that has a value of None"
            )));
        }

        self.storage[idx].take();
        self.empty_idx.push(idx);

        Ok(())
    }
}
