use std::{
    cell::RefCell,
    ops::{Mul, SubAssign},
    rc::Rc,
};

use crate::prelude::*;

pub struct Sgd<F> {
    lr: F,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
}

impl<F> Sgd<F> {
    pub fn new(lr: F, module: &Module<F>) -> Sgd<F> {
        Self {
            lr,
            storage: module.storage.clone(),
        }
    }

    pub fn optim(&self) -> Result<(), PzeudoErr>
    where
        F: Mul<Output = F> + Copy + SubAssign,
    {
        for permanent in self.storage.borrow_mut().get_mut_permanent_storage() {
            permanent
                .array
                .sub_assign(&permanent.grad.mul_scalar(self.lr)?)?;
        }
        Ok(())
    }
}
