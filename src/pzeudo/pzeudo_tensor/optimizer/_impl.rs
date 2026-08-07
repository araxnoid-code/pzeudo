use crate::prelude::*;
use serde::Serialize;

impl<F> SaveParamsTrait<F> for Sgd<F>
where
    F: Serialize,
{
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}

impl<F> SaveParamsTrait<F> for SgdMomentum<F>
where
    F: Serialize,
{
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}

impl<F> SaveParamsTrait<F> for AdaGrad<F>
where
    F: Serialize,
{
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}

impl<F> SaveParamsTrait<F> for RMSProp<F>
where
    F: Serialize,
{
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}

impl<F> SaveParamsTrait<F> for Adam<F>
where
    F: Serialize,
{
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}
