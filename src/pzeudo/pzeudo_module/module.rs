use crate::prelude::*;
use std::{cell::RefCell, rc::Rc};

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
}
