use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum RecordStatus {
    Record(usize),
    UnRecord(usize),
}

impl<F, G> Tensor<F, Contiguous, G> {
    pub fn unrecord(&mut self) -> Result<(), PzeudoErr> {
        let mut record = self.record.borrow_mut();
        if let RecordStatus::Record(idx) = self.record_status.ok_or(PzeudoErr::TensorErr(
            format!("Tensor::unrecord. The tensor has no status record."),
        ))? {
            *record
                .skip
                .get_mut(idx)
                .ok_or(PzeudoErr::TensorErr(format!(
                    "Tensor::unrecord. Index {idx} points to an invalid location in the record.(skip)"
                )))? = true;

            self.record_status = Some(RecordStatus::UnRecord(idx));
        } else {
            return Err(PzeudoErr::TensorErr(format!(
                "Tensor::unrecord. tensor with UnRecord status"
            )));
        }

        Ok(())
    }

    pub fn record(&mut self) -> Result<(), PzeudoErr> {
        let mut record = self.record.borrow_mut();
        if let RecordStatus::UnRecord(idx) = self.record_status.ok_or(PzeudoErr::TensorErr(
            format!("Tensor::record. The tensor has no status record."),
        ))? {
            *record
                .skip
                .get_mut(idx)
                .ok_or(PzeudoErr::TensorErr(format!(
                    "Tensor::record. Index {idx} points to an invalid location in the record.(skip)"
                )))? = false;

            self.record_status = Some(RecordStatus::Record(idx));
        } else {
            return Err(PzeudoErr::TensorErr(format!(
                "Tensor::record. tensor with Record status"
            )));
        }

        Ok(())
    }
}
