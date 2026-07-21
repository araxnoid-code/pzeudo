use crate::prelude::*;
use std::{cell::RefCell, rc::Rc};

pub struct Module<F> {
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>,
}
