use num_traits::Zero;

use crate::prelude::*;
use std::{cell::RefCell, ops::MulAssign, rc::Rc};

pub struct Module<F> {
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>,
}

impl<F> Module<F> {
    pub fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }

    pub fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>> {
        &self.record
    }

    pub fn clear_storage(&self) {
        self.storage.borrow_mut().clear_storage();
    }

    pub fn zero_grad(&self)
    where
        F: Zero,
    {
        for permanent in self.storage.borrow_mut().get_mut_permanent_storage() {
            permanent.grad.to_zero();
        }
    }
}
