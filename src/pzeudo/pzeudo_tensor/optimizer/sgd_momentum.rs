use crate::prelude::*;
use num_traits::{Float, Zero};
use std::{
    cell::RefCell,
    ops::{AddAssign, Mul, MulAssign, SubAssign},
    rc::Rc,
};

/// ## SGD + Momentum
/// - w_new = w_old - v_new
/// - v_new = hyperparameter * v_old + lr * grad(w_old)
/// - mu = 0.9 (default). Modify via SgdMomentum::set_mu
pub struct SgdMomentum<F> {
    lr: F,
    pub(crate) v: Vec<Array<F>>,
    pub(crate) hyperparameter: F,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) range: (usize, usize),
}

impl<F> SgdMomentum<F>
where
    F: Float,
{
    pub fn new(lr: F, mut model_builder: ModelBuilder<F>) -> Result<SgdMomentum<F>, PzeudoErr> {
        if let Some(load_params) = &model_builder.load_params {
            if !load_params.is_empty() {
                return Err(PzeudoErr::OptimErr(format!(
                    "SgdMomentum::new. Load Params in ModelBuilder are not all used, identifying the Model architecture as not being the same as the stored parameters."
                )));
            }
        }

        let start = model_builder.start;
        let module = model_builder.get_module();
        let storage = module.storage.borrow();
        let params_storage = &storage.get_params_storage().storage;
        let mut vec = Vec::with_capacity(params_storage.len());
        for param in params_storage {
            vec.push(Array::<F>::zeros(&param.array.shape));
        }

        Ok(Self {
            lr,
            range: (start, start + vec.len()),
            v: vec,
            hyperparameter: F::from(0.9).ok_or(PzeudoErr::OpsErr(format!(
                "SgdMomentum::new. Unable to cast the default momentum (0.9) data type."
            )))?,
            storage: module.storage.clone(),
        })
    }

    pub fn set_lr(&mut self, lr: F) {
        self.lr = lr;
    }

    pub fn set_hyperparameter(&mut self, hyperparameter: F) {
        self.hyperparameter = hyperparameter;
    }

    /// ## formula:
    /// - w_new = w_old - v_new
    /// - v_new = hyperparameter * v_old + lr * grad(w_old)
    /// - mu = 0.9 (default). Modify via SgdMomentum::set_mu
    pub fn optim(&mut self) -> Result<(), PzeudoErr>
    where
        F: Mul<Output = F> + Copy + SubAssign + MulAssign + AddAssign,
    {
        for (idx, param) in &mut self.storage.borrow_mut().get_params_storage_mut().storage
            [self.range.0..self.range.1]
            .iter_mut()
            .enumerate()
        {
            if let Some(grad) = &param.grad {
                let v_arr = self.v.get_mut(idx).ok_or(PzeudoErr::OptimErr(format!("SgdMomentum::optim. Index {idx} points to an invalid location in the v (velocity) list.")))?;
                let len = v_arr.shape.iter().product::<usize>();
                for i in 0..len {
                    let x = v_arr.linear_index_mut(i)?;
                    *x *= self.hyperparameter;
                    *x += self.lr * grad.linear_index(i)?;
                    *param.array.linear_index_mut(i)? -= *x;
                }
            }
        }
        Ok(())
    }

    /// will set all gradients in storage params to 0.
    pub fn zero_grad(&self)
    where
        F: Zero,
    {
        for permanent in &mut self.storage.borrow_mut().get_params_storage_mut().storage
            [self.range.0..self.range.1]
        {
            if let Some(grad) = &mut permanent.grad {
                grad.to_zeros();
            }
        }
    }
}
