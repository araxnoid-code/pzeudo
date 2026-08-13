use crate::prelude::*;
use rand::rngs::SmallRng;
use std::{cell::RefCell, rc::Rc};

pub struct Module<F, M> {
    pub(crate) model: Option<M>,
    pub(crate) builder: ModuleBuilder<F>,
}

impl<F, M> Module<F, M> {
    pub fn new(model: M, builder: ModuleBuilder<F>) -> Module<F, M> {
        Module {
            model: Some(model),
            builder,
        }
    }

    pub fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.builder.storage
    }

    pub fn get_record(&self) -> &Rc<RefCell<Record<F>>> {
        &self.builder.record
    }

    pub fn get_seed(&self) -> u64 {
        self.builder.seed
    }

    pub fn get_rng_mut(&mut self) -> &mut SmallRng {
        &mut self.builder.rng
    }

    pub fn clear_storage(&self) {
        self.builder.clear_storage();
    }
}
