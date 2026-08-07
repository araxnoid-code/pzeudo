use crate::prelude::*;
use num_traits::{Float, Zero};
use std::{
    cell::RefCell,
    ops::{AddAssign, Mul, MulAssign, SubAssign},
    rc::Rc,
};

/// ## RMSProp
/// - w_new = w_old - lr/√(g_new + eps) * grad(w_old)
/// - g_new = hyperparameter * g_old + (1 - hyperparameter)  * grad(w_old)^2
/// - hyperparameter = 0.9(default). Modify via RMSProp::set hyperparameter.
/// - eps = 1e-7
pub struct RMSProp<F> {
    lr: F,
    pub(crate) g: Vec<Array<F>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) hyperparameter: F,
    pub(crate) range: (usize, usize),
}

impl<F> RMSProp<F>
where
    F: Float,
{
    pub fn new(lr: F, mut model_builder: ModelBuilder<F>) -> Result<RMSProp<F>, PzeudoErr> {
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
            g: vec,
            storage: module.storage.clone(),
            hyperparameter: F::from(0.9).ok_or(PzeudoErr::OpsErr(format!(
                "RMSProp::new. Unable to cast the default momentum (0.9) data type."
            )))?,
        })
    }

    pub fn set_lr(&mut self, lr: F) {
        self.lr = lr;
    }

    pub fn set_hyperparameter(&mut self, hyperparameter: F) {
        self.hyperparameter = hyperparameter;
    }

    /// formula:
    /// - w_new = w_old - lr/√(g_new + eps) * grad(w_old)
    /// - g_new = hyperparameter * g_old + (1 - hyperparameter)  * grad(w_old)^2
    /// - hyperparameter = 0.9(default). Modify via RMSProp::set hyperparameter.
    /// - eps = 1e-7
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
                    "RMSProp::optim. Index {idx} points to an invalid location in the g list."
                )))?;
                let epsilon = F::from(1e-7).ok_or(PzeudoErr::OptimErr(format!(
                    "RMSProp::optim. Unable to cast data type for epsilon 1e-7."
                )))?;
                let one = F::one();

                let len = g_arr.shape.iter().product::<usize>();
                for i in 0..len {
                    let g = g_arr.linear_index_mut(i)?;
                    let grad = grad.linear_index(i)?;

                    *g *= self.hyperparameter;
                    *g += (one - self.hyperparameter) * (grad * grad);

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
