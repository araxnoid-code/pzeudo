use std::{
    cell::RefCell,
    ops::{Mul, SubAssign},
    rc::Rc,
};

use num_traits::Zero;

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
        for permanent in &mut self.storage.borrow_mut().get_params_storage_mut().storage {
            if let Some(grad) = &permanent.grad {
                permanent.array.sub_assign(&grad.mul_scalar(self.lr)?)?;
            }
        }
        Ok(())
    }

    pub fn zero_grad(&self)
    where
        F: Zero,
    {
        for permanent in &mut self.storage.borrow_mut().get_params_storage_mut().storage {
            if let Some(grad) = &mut permanent.grad {
                grad.to_zeros();
            }
        }
    }
}
