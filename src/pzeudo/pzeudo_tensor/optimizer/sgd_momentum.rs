use crate::prelude::*;
use num_traits::{Float, Zero};
use std::{
    cell::RefCell,
    ops::{AddAssign, Mul, MulAssign, SubAssign},
    rc::Rc,
};

/// ## SGD + Momentum
/// - w_new = w_old - v_new
/// - v_new = mu * v_old + lr * grad(w_old)
/// - mu = 0.9 (default). can be changed via SgdMomentum::set_mu
pub struct SgdMomentum<F> {
    lr: F,
    pub(crate) v: Vec<Array<F>>,
    pub(crate) mu: F,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
}

impl<F> SgdMomentum<F>
where
    F: Float,
{
    pub fn new(lr: F, module: &Module<F>) -> Result<SgdMomentum<F>, PzeudoErr> {
        let storage = module.storage.borrow();
        let params_storage = &storage.get_params_storage().storage;
        let mut vec = Vec::with_capacity(params_storage.len());
        for param in params_storage {
            vec.push(Array::<F>::zeros(&param.array.shape));
        }

        Ok(Self {
            lr,
            v: vec,
            mu: F::from(0.9).ok_or(PzeudoErr::OpsErr(format!(
                "SgdMomentum::new. Unable to cast the default momentum (0.9) data type."
            )))?,
            storage: module.storage.clone(),
        })
    }

    pub fn set_lr(&mut self, lr: F) {
        self.lr = lr;
    }

    pub fn set_mu(&mut self, mu: F) {
        self.mu = mu;
    }

    /// ### formula:
    /// - w_new = w_old - v_new
    /// - v_new = mu * v_old + lr * grad(w_old)
    /// - mu = 0.9 (default). can be changed via SgdMomentum::set_mu
    pub fn optim(&mut self) -> Result<(), PzeudoErr>
    where
        F: Mul<Output = F> + Copy + SubAssign + MulAssign + AddAssign,
    {
        for (idx, permanent) in &mut self
            .storage
            .borrow_mut()
            .get_params_storage_mut()
            .storage
            .iter_mut()
            .enumerate()
        {
            if let Some(grad) = &permanent.grad {
                let v_arr = self.v.get_mut(idx).ok_or(PzeudoErr::OptimErr(format!("SgdMomentum::optim. Index {idx} points to an invalid location in the v (velocity) list.")))?;
                let len = v_arr.shape.iter().product::<usize>();
                for i in 0..len {
                    let x = v_arr.linear_index_mut(i)?;
                    *x *= self.mu;
                    *x += self.lr * grad.linear_index(i)?;
                }

                permanent.array.sub_assign(v_arr)?;
            }
        }
        Ok(())
    }

    /// will set all gradients in storage params to 0.
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
