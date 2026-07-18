use crate::prelude::*;

pub struct ArrayStorage<F> {
    storage: Vec<Option<Vec<F>>>,
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

    pub fn get_storage(&self) -> &Vec<Option<Vec<F>>> {
        &self.storage
    }

    pub fn get_mut_storage(&mut self) -> &mut Vec<Option<Vec<F>>> {
        &mut self.storage
    }

    pub fn get_empty_idx(&self) -> &Vec<usize> {
        &self.empty_idx
    }

    pub fn get_mut_empty_idx(&mut self) -> &mut Vec<usize> {
        &mut self.empty_idx
    }

    pub fn push(&mut self, element: Vec<F>) -> Result<usize, PzeudoErr> {
        let idx = if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoErr::StoragePushErr(format!(
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

    pub fn remove(&mut self, idx: usize) -> Result<(), PzeudoErr> {
        if self.storage[idx].is_none() {
            return Err(PzeudoErr::StorageRemoveErr(format!(
                "ArrayStorage::remove. index {idx} points to an element that has a value of None"
            )));
        }

        self.storage[idx].take();
        self.empty_idx.push(idx);

        Ok(())
    }

    pub fn get_element(&self, idx: usize) -> Result<&Vec<F>, PzeudoErr> {
        let data = self
            .storage
            .get(idx)
            .ok_or(PzeudoErr::StorageGetErr(format!(
                "ArrayStorage::get. index {idx} points to an invalid location on storage."
            )))?
            .as_ref()
            .ok_or(PzeudoErr::StorageGetErr(format!(
                "ArrayStorage::get. index {idx} points to elements that have the value None in storage."
            )))?;

        Ok(data)
    }
}
