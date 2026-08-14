use crate::prelude::*;
use num_traits::Float;
use std::{
    cell::RefCell,
    ops::{AddAssign, Mul, MulAssign, SubAssign},
    rc::Rc,
};

/// ## Adam
/// - w_new = w_old - lr/√(g_hat + eps) * m_hat
/// - g_new = hyperparameter_g * g_old + (1 - hyperparameter_g)  * grad(w_old)^2
/// - g_hat = g_new/(1 - hyperparameter_g^i)
/// - m_new = hyperparameter_m * m_old + (1 - hyperparameter_m) * grad(w_old)
/// - m_hat = m_new/(1 - hyperparameter_m^i)
/// - hyperparameter_m = 0.9(default) Modify via Adam::set_hyperparameter_m
/// - hyperparameter_g = 0.99(default) Modify via Adam::set_hyperparameter_g
/// - i = iteration
/// - eps = 1e-7
pub struct Adam<F> {
    lr: F,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) g: Vec<Array<F>>,
    pub(crate) m: Vec<Array<F>>,
    pub(crate) hyperparameter_g: F,
    pub(crate) hyperparameter_m: F,
    pub(crate) range: (usize, usize),
    t: i32,
}

impl<F> Adam<F>
where
    F: Float,
{
    pub fn new(lr: F, mut model_builder: ModelBuilder<F>) -> Result<Adam<F>, PzeudoErr> {
        if let Some(load_params) = &model_builder.load_params {
            if !load_params.is_empty() {
                return Err(PzeudoErr::OptimErr(format!(
                    "Adam::new. Load Params in ModelBuilder are not all used, identifying the Model architecture as not being the same as the stored parameters."
                )));
            }
        }

        let start = model_builder.start;
        let module = model_builder.get_module();
        let storage = module.storage.borrow();
        let params_storage = &storage.get_params_storage().storage;

        let mut vec_g = Vec::with_capacity(params_storage.len());
        let mut vec_m = Vec::with_capacity(params_storage.len());
        for param in params_storage {
            vec_g.push(Array::<F>::zeros(&param.array.shape));
            vec_m.push(Array::<F>::zeros(&param.array.shape));
        }

        Ok(Self {
            lr,
            range: (start, start + vec_g.len()),
            g: vec_g,
            m: vec_m,
            storage: module.storage.clone(),
            hyperparameter_g: F::from(0.99).ok_or(PzeudoErr::OpsErr(format!(
                "Adam::new. Unable to cast the default momentum (0.99) data type."
            )))?,
            hyperparameter_m: F::from(0.9).ok_or(PzeudoErr::OpsErr(format!(
                "Adam::new. Unable to cast the default momentum (0.9) data type."
            )))?,
            t: 1,
        })
    }

    pub fn set_lr(&mut self, lr: F) {
        self.lr = lr;
    }

    /// - hyperparameter_g = 0.99(default)
    pub fn set_hyperparameter_g(&mut self, hyperparameter_g: F) {
        self.hyperparameter_g = hyperparameter_g;
    }

    /// - hyperparameter_m = 0.9(default)
    pub fn set_hyperparameter_m(&mut self, hyperparameter_m: F) {
        self.hyperparameter_m = hyperparameter_m;
    }

    /// ## formula:
    /// - w_new = w_old - lr/√(g_hat + eps) * m_hat
    /// - g_new = hyperparameter_g * g_old + (1 - hyperparameter_g)  * grad(w_old)^2
    /// - g_hat = g_new/(1 - hyperparameter_g^i)
    /// - m_new = hyperparameter_m * m_old + (1 - hyperparameter_m) * grad(w_old)
    /// - m_hat = m_new/(1 - hyperparameter_m^i)
    /// - hyperparameter_m = 0.9(default) Modify via Adam::set_hyperparameter_m
    /// - hyperparameter_g = 0.99(default) Modify via Adam::set_hyperparameter_g
    /// - i = iteration
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
                    "Adam::optim. Index {idx} points to an invalid location in the g list."
                )))?;

                let m_arr = self.m.get_mut(idx).ok_or(PzeudoErr::OptimErr(format!(
                    "Adam::optim. Index {idx} points to an invalid location in the m list."
                )))?;

                let epsilon = F::from(1e-7).ok_or(PzeudoErr::OptimErr(format!(
                    "Adam::optim. Unable to cast data type for epsilon 1e-7."
                )))?;
                let one = F::one();

                let len = g_arr.shape.iter().product::<usize>();
                for i in 0..len {
                    let grad = grad.linear_index(i)?;
                    // m
                    let m = m_arr.linear_index_mut(i)?;
                    *m *= self.hyperparameter_m;
                    *m += (one - self.hyperparameter_m) * grad;
                    let m_hat = *m / (one - self.hyperparameter_m.powi(self.t));

                    // g
                    let g = g_arr.linear_index_mut(i)?;
                    *g *= self.hyperparameter_g;
                    *g += (one - self.hyperparameter_g) * (grad * grad);
                    let g_hat = *g / (one - self.hyperparameter_g.powi(self.t));

                    // update
                    *param.array.linear_index_mut(i)? -= self.lr / (g_hat + epsilon).sqrt() * m_hat;
                }
            }
        }

        self.t += 1;
        Ok(())
    }
}
