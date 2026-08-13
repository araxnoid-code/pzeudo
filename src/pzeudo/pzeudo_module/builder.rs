use crate::{prelude::*, pzeudo_module::module::Module};
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::RefCell, rc::Rc};

pub struct ModuleBuilder<F> {
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) record: Rc<RefCell<Vec<Option<RecordLabel<F>>>>>,
    pub(crate) rng: SmallRng,
    pub(crate) seed: u64,
}

impl<F> ModuleBuilder<F> {
    pub fn new(seed: u64) -> ModuleBuilder<F> {
        let storage = Rc::new(RefCell::new(ArrayStorage::<F>::new(None)));
        let record = Rc::new(RefCell::new(Vec::new()));
        let rng = rand::rngs::SmallRng::seed_from_u64(seed);

        Self {
            record,
            storage,
            rng,
            seed,
        }
    }

    /// will create a ModelBuilder, needed to build the model architecture and optimizer.
    pub fn model_builder(&mut self) -> ModelBuilder<'_, F> {
        ModelBuilder::new(self)
    }

    pub fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }

    pub fn get_record(&self) -> &Rc<RefCell<Vec<Option<RecordLabel<F>>>>> {
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

    pub fn build<M>(self, model: M) -> Module<F, M> {
        Module::new(model, self)
    }
}
