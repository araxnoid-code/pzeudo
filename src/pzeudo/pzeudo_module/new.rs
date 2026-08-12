use std::{cell::RefCell, rc::Rc};

use rand::SeedableRng;

use crate::prelude::*;

impl<F> Module<F> {
    pub fn new(seed: u64) -> Module<F> {
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
}
