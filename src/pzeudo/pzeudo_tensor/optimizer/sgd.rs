use crate::prelude::*;
use std::{
    cell::RefCell,
    ops::{Mul, SubAssign},
    rc::Rc,
};

/// ## SGD (Stochastic Gradient Descent)
/// w_new = w_old - lr * grad(w_old)
pub struct Sgd<F> {
    lr: F,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) range: (usize, usize),
}

impl<F> Sgd<F> {
    pub fn new(lr: F, mut model_builder: ModelBuilder<F>) -> Result<Sgd<F>, PzeudoErr> {
        if let Some(load_params) = &model_builder.load_params {
            if !load_params.is_empty() {
                return Err(PzeudoErr::OptimErr(format!(
                    "Sgd::new. Load Params in ModelBuilder are not all used, identifying the Model architecture as not being the same as the stored parameters."
                )));
            }
        }

        let start = model_builder.start;
        let module = model_builder.get_module();
        Ok(Self {
            lr,
            range: (
                start,
                start
                    + module
                        .get_storage()
                        .borrow()
                        .get_params_storage()
                        .storage
                        .len(),
            ),
            storage: module.storage.clone(),
        })
    }

    pub fn set_lr(&mut self, lr: F) {
        self.lr = lr;
    }

    /// ## formula:
    /// w_new = w_old - lr * grad(w_old)
    pub fn optim(&self) -> Result<(), PzeudoErr>
    where
        F: Mul<Output = F> + Copy + SubAssign,
    {
        for param in &mut self.storage.borrow_mut().get_params_storage_mut().storage
            [self.range.0..self.range.1]
        {
            if let Some(grad) = &param.grad {
                param.array.sub_assign(&grad.mul_scalar(self.lr)?)?;
            }
        }
        Ok(())
    }
}
