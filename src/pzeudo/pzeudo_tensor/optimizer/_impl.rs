use crate::prelude::*;
use serde::Serialize;

impl<F> OptimizerTrait<F> for Sgd<F> {
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}
impl<F> SaveParamsTrait<F> for Sgd<F> where F: Serialize {}

impl<F> OptimizerTrait<F> for SgdMomentum<F> {
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}
impl<F> SaveParamsTrait<F> for SgdMomentum<F> where F: Serialize {}

impl<F> OptimizerTrait<F> for AdaGrad<F> {
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}
impl<F> SaveParamsTrait<F> for AdaGrad<F> where F: Serialize {}

impl<F> OptimizerTrait<F> for RMSProp<F> {
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}
impl<F> SaveParamsTrait<F> for RMSProp<F> where F: Serialize {}

impl<F> OptimizerTrait<F> for Adam<F> {
    fn get_storage(&self) -> &std::rc::Rc<std::cell::RefCell<ArrayStorage<F>>> {
        &self.storage
    }
    fn get_range(&self) -> (usize, usize) {
        self.range
    }
}
impl<F> SaveParamsTrait<F> for Adam<F> where F: Serialize {}
