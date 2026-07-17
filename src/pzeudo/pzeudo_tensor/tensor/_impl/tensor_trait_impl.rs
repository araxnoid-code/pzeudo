use std::{cell::RefCell, rc::Rc};

use crate::prelude::*;

impl<F, T> TensorTrait<F> for Tensor<F, T> {
    fn get_array_idx(&self) -> usize {
        self.array_idx
    }

    fn get_array_metadata(&self) -> &crate::prelude::TensorMetadata {
        &self.array_metadata
    }

    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>> {
        &self.record
    }

    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }
}
