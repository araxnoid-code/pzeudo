use std::{cell::RefCell, rc::Rc};

use crate::prelude::*;

pub trait TensorTrait<F> {
    fn get_array_idx(&self) -> StorageType;
    fn get_grad_idx(&self) -> Option<StorageType>;
    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>>;
    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>>;
}

impl<F, T> TensorTrait<F> for Tensor<F, T> {
    fn get_array_idx(&self) -> StorageType {
        self.array_idx
    }

    fn get_grad_idx(&self) -> Option<StorageType> {
        self.grad_idx
    }

    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>> {
        &self.record
    }

    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }
}
