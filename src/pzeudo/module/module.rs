use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{ModuleInit, StorageTrait};

pub struct Module<GradStorage, Grad, Record>
where
    GradStorage: StorageTrait<Grad>,
{
    pub grad_storage: Rc<RefCell<GradStorage>>,
    pub record_storage: Rc<RefCell<Vec<(Record, Option<usize>)>>>,
    _type_phantom: PhantomData<Grad>,
}

impl<GradStorage, Grad, Record> Module<GradStorage, Grad, Record>
where
    GradStorage: StorageTrait<Grad>,
{
    pub fn new(init: ModuleInit<GradStorage, Record>) -> Module<GradStorage, Grad, Record> {
        Self {
            grad_storage: Rc::new(RefCell::new(init.0)),
            record_storage: Rc::new(RefCell::new(Vec::new())),
            _type_phantom: Default::default(),
        }
    }
}
