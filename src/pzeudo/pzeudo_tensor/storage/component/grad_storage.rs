use crate::prelude::*;

pub struct GradStorage<F> {
    is_update: Vec<bool>,
    status: Vec<u8>,
    // status:
    //      0: remove
    //      1: no_grad
    //      2: Value
    time: Vec<usize>,
    storage: Vec<Option<Array<F>>>,
    empty_idx: Vec<usize>,
}

impl<F> GradStorage<F> {
    pub(crate) fn new(capacity: Option<usize>) -> GradStorage<F> {
        Self {
            is_update: Vec::with_capacity(capacity.unwrap_or(0)),
            status: Vec::with_capacity(capacity.unwrap_or(0)),
            time: Vec::with_capacity(capacity.unwrap_or(0)),
            storage: Vec::with_capacity(capacity.unwrap_or(0)),
            empty_idx: Vec::new(),
        }
    }

    pub(crate) fn take_grad(&mut self, idx: usize) -> Result<Array<F>, PzeudoErr> {
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

    pub(crate) fn replace_grad(&mut self, idx: usize, grad: Array<F>) -> Result<(), PzeudoErr> {
        let space = self.storage.get_mut(idx).ok_or(PzeudoErr::StorageErr(format!(
            "GradStorage::replace_grad. index {idx} points to an invalid location on gradient storage."
        )))?;

        if space.is_some() {
            return Err(PzeudoErr::StorageErr(format!(
                "GradStorage::replace_grad. Index {idx} points to the location where the gradient is stored in gradient storage."
            )));
        }

        space.replace(grad);
        Ok(())
    }

    pub(crate) fn push_grad(&mut self, array: Array<F>) -> Result<(usize, usize), PzeudoErr> {
        if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoErr::StorageErr(format!(
                    "GradStorage::push_grad. The problem occurs because the index {idx} obtained from empty_idx points to an element that still has a value."
                )));
            }
            self.storage[idx] = Some(array);
            self.status[idx] = 2;
            self.time[idx] += 1;
            self.is_update[idx] = false;

            return Ok((idx, self.time[idx]));
        } else {
            self.status.push(2);
            self.time.push(0);
            self.storage.push(Some(array));
            self.is_update.push(false);
            return Ok((self.storage.len() - 1, 0));
        };
    }

    pub(crate) fn get_grad(&self, idx: usize, grad_time: usize) -> Result<&Array<F>, PzeudoErr> {
        let status = self.status
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::get_grad. index {idx} points to an invalid location on gradient storage(status).")))?;

        let time = self.time
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::get_grad. index {idx} points to an invalid location on gradient storage(time).")))?;

        if *status == 0 {
            return Err(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to elements that have the value None in gradient storage(status)."
            )));
        } else if *status == 1 {
            return Err(PzeudoErr::StorageNoGradErr(format!(
                "GradStorage::get_grad. index {idx} points to elements that have the value None in gradient storage because the gradient is set to no_grad(status)."
            )));
        } else if *time != grad_time {
            return Err(PzeudoErr::StorageTimeErr(format!(
                "GradStorage::get_grad. index {idx} points to an element that has a different time value. time owned by {grad_time}, time on element {time}(time)."
            )));
        }

        let grad = self
            .storage
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to an invalid location on gradient storage(storage)."
            )))?
            .as_ref()
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to elements that have the value None in gradient storage(storage)."
            )))?;

        Ok(grad)
    }

    pub(crate) fn get_grad_mut(
        &mut self,
        idx: usize,
        grad_time: usize,
    ) -> Result<&mut Array<F>, PzeudoErr> {
        let status = self.status
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::get_grad_mut. index {idx} points to an invalid location on gradient storage(status).")))?;

        let time = self.time
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::get_grad_mut. index {idx} points to an invalid location on gradient storage(time).")))?;

        if *status == 0 {
            return Err(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad_mut. index {idx} points to elements that have the value None in gradient storage(status)."
            )));
        } else if *status == 1 {
            return Err(PzeudoErr::StorageNoGradErr(format!(
                "GradStorage::get_grad_mut. index {idx} points to elements that have the value None in gradient storage because the gradient is set to no_grad(status)."
            )));
        } else if *time != grad_time {
            return Err(PzeudoErr::StorageTimeErr(format!(
                "GradStorage::get_grad_mut. index {idx} points to an element that has a different time value. time owned by {grad_time}, time on element {time}(time)."
            )));
        }

        let grad = self
            .storage
            .get_mut(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to an invalid location on gradient storage."
            )))?
            .as_mut()
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::get_grad. index {idx} points to elements that have the value None in gradient storage."
            )))?;

        Ok(grad)
    }

    pub fn clear(&mut self) {
        self.status.clear();
        self.time.clear();
        self.storage.clear();
        self.empty_idx.clear();
        self.is_update.clear();
    }

    pub fn is_update(&self, idx: usize) -> Result<bool, PzeudoErr> {
        let status = self.is_update
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::is_update. index {idx} points to an invalid location on gradient storage(status).")))?;

        Ok(*status)
    }

    pub fn is_no_grad(&self, idx: usize) -> Result<bool, PzeudoErr> {
        let status = self.status
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::check_no_grad. index {idx} points to an invalid location on gradient storage(status).")))?;

        if *status == 0 {
            return Err(PzeudoErr::StorageErr(format!(
                "GradStorage::check_no_grad. index {idx} points to elements that have the value None in gradient storage(status)."
            )));
        } else if *status == 1 {
            return Ok(true);
        } else {
            return Ok(false);
        }
    }

    pub fn is_time_not_match(&self, idx: usize, grad_time: usize) -> Result<bool, PzeudoErr> {
        let time = self.time
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!("GradStorage::get_grad_mut. index {idx} points to an invalid location on gradient storage(time).")))?;

        if *time != grad_time {
            return Ok(true);
        }

        Ok(false)
    }

    fn _remove_grad(&mut self, idx: usize) -> Result<(), PzeudoErr> {
        self
            .storage
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::remove_grad. index {idx} points to an invalid location on gradient storage.")))?
            .as_ref()
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::remove_grad. index {idx} points to elements that have the value None in gradient storage."
        )))?;

        self.status[idx] = 0;
        self.time[idx] += 1;
        self.storage[idx] = None;
        self.empty_idx.push(idx);

        Ok(())
    }

    pub fn no_grad(&mut self, idx: usize, grad_time: usize) -> Result<(), PzeudoErr> {
        let time = self.time.get(idx).ok_or(PzeudoErr::StorageErr(format!(
            "GradStorage::no_grad. index {idx} points to an invalid location on gradient storage(time).")))?;

        if *time != grad_time {
            return Err(PzeudoErr::StorageTimeErr(format!(
                "GradStorage::no_grad. index {idx} points to an element that has a different time value. time owned by {grad_time}, time on element {time}(time)."
            )));
        }

        self
            .storage
            .get(idx)
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::no_grad. index {idx} points to an invalid location on gradient storage.")))?
            .as_ref()
            .ok_or(PzeudoErr::StorageErr(format!(
                "GradStorage::no_grad. index {idx} points to elements that have the value None in gradient storage."
        )))?;

        self.status[idx] = 1;
        self.time[idx] += 1;
        self.storage[idx] = None;
        self.empty_idx.push(idx);

        Ok(())
    }
}
