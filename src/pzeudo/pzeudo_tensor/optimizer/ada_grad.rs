use crate::prelude::*;
use num_traits::{Float, Zero};
use std::{
    cell::RefCell,
    ops::{AddAssign, Mul, MulAssign, SubAssign},
    rc::Rc,
};

/// ## AdaGrad
/// - w_new = w_old - lr/√(g_new + e) * grad(w_old)
/// - g_new = g_old * grad(w_old)^2
pub struct AdaGrad<F> {
    lr: F,
    pub(crate) g: Vec<Array<F>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) range: (usize, usize),
}

impl<F> AdaGrad<F>
where
    F: Float,
{
    pub fn new(lr: F, mut model_builder: ModelBuilder<F>) -> Result<AdaGrad<F>, PzeudoErr> {
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
            range: (start, vec.len()),
            g: vec,
            storage: module.storage.clone(),
        })
    }

    pub fn set_lr(&mut self, lr: F) {
        self.lr = lr;
    }

    /// formula:
    /// - w_new = w_old - lr/√(g_new + e) * grad(w_old)
    /// - g_new = g_old * grad(w_old)^2
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
                let g_arr = self.g.get_mut(idx).ok_or(PzeudoErr::OptimErr(format!(
                    "AdaGrad::optim. Index {idx} points to an invalid location in the g list."
                )))?;
                let epsilon = F::from(1e-7).ok_or(PzeudoErr::OptimErr(format!(
                    "AdaGrad::optim. Unable to cast data type for epsilon 1e-7."
                )))?;

                let len = g_arr.shape.iter().product::<usize>();
                for i in 0..len {
                    let g = g_arr.linear_index_mut(i)?;
                    let grad = grad.linear_index(i)?;
                    *g += grad * grad;

                    let update = self.lr / (*g + epsilon).sqrt() * grad;
                    *param.array.linear_index_mut(i)? -= update;
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
