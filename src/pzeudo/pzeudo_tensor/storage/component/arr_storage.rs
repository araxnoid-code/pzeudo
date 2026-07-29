use std::format;

use crate::prelude::*;

pub struct ArrStorage<F> {
    storage: Vec<Option<Array<F>>>,
    empty_idx: Vec<usize>,
}

impl<F> ArrStorage<F> {
    pub(crate) fn new(capacity: Option<usize>) -> ArrStorage<F> {
        Self {
            storage: Vec::with_capacity(capacity.unwrap_or(0)),
            empty_idx: Vec::new(),
        }
    }

    pub(crate) fn push_arr(&mut self, element: Array<F>) -> Result<usize, PzeudoErr> {
        if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoErr::ArrStoragePushErr(format!(
                    "ArrStorage::arr_push. The problem occurs because the index {idx} obtained from empty_idx points to an element that still has a value."
                )));
            }

            return Ok(idx);
        } else {
            self.storage.push(Some(element));
            return Ok(self.storage.len() - 1);
        };
    }

    pub(crate) fn get_arr(&self, idx: usize) -> Result<&Array<F>, PzeudoErr> {
        let array = self
            .storage
            .get(idx)
            .ok_or(PzeudoErr::ArrStorageGetErr(format!(
                "ArrStorage::get_arr. index {idx} points to an invalid location on arr storage."
            )))?
            .as_ref()
            .ok_or(PzeudoErr::ArrStorageGetErr(format!(
                "ArrStorage::get_arr. index {idx} points to elements that have the value None in arr storage."
            )))?;

        Ok(array)
    }

    pub(crate) fn get_arr_mut(&mut self, idx: usize) -> Result<&mut Array<F>, PzeudoErr> {
        let array = self
            .storage
            .get_mut(idx)
            .ok_or(PzeudoErr::ArrStorageGetMutErr(format!(
                "ArrStorage::get_arr_mut. index {idx} points to an invalid location on arr storage."
            )))?
            .as_mut()
            .ok_or(PzeudoErr::ArrStorageGetMutErr(format!(
                "ArrStorage::get_arr_mut. index {idx} points to elements that have the value None in arr storage."
            )))?;

        Ok(array)
    }

    pub fn clear(&mut self) {
        self.storage.clear();
        self.empty_idx.clear();
    }
}

// impl<F> Deref for ArrStorage<F> {
//     type Target = Vec<Option<Array<F>>>;
//     fn deref(&self) -> &Self::Target {
//         &self.storage
//     }
// }

// impl<F> DerefMut for ArrStorage<F> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.storage
//     }
// }
