use num_traits::Zero;

use crate::prelude::*;
use std::{cell::RefCell, rc::Rc};

pub trait OptimizerTrait<F> {
    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>>;
    fn get_range(&self) -> (usize, usize);

    /// will set all gradients in storage params to zero
    fn zero_grad(&self)
    where
        F: Zero,
    {
        let range = self.get_range();
        let mut storage_borrow_mut = self.get_storage().borrow_mut();
        let params_storage = storage_borrow_mut.get_params_storage_mut();

        for permanent in &mut params_storage.storage[range.0..range.1] {
            if let Some(grad) = &mut permanent.grad {
                grad.to_zeros();
            }
        }

        for update_stat in &mut storage_borrow_mut.get_params_storage_mut().update {
            *update_stat = false;
        }
    }
}
