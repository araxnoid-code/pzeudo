use rand::rngs::SmallRng;

use crate::prelude::*;
use std::{cell::RefCell, rc::Rc};

pub struct Module<F> {
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) record: Rc<RefCell<Vec<RecordLabel<F>>>>,
    pub(crate) rng: SmallRng,
    pub(crate) seed: u64,
}

impl<F> Module<F> {
    pub fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }

    pub fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel<F>>>> {
        &self.record
    }

    pub fn get_seed(&self) -> u64 {
        self.seed
    }

    pub fn get_rng_mut(&mut self) -> &mut SmallRng {
        &mut self.rng
    }

    pub fn clear_storage(&self) {
        self.storage.borrow_mut().clear_storage();
    }
}
