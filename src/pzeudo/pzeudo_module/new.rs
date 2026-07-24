use std::{cell::RefCell, rc::Rc};

use crate::prelude::*;

impl<F> Module<F> {
    pub fn new() -> Module<F> {
        let storage = Rc::new(RefCell::new(ArrayStorage::<F>::new(None)));
        let record = Rc::new(RefCell::new(Vec::<RecordLabel>::new()));

        Self { record, storage }
    }
}
