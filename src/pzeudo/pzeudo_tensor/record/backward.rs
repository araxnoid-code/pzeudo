use std::ops::{AddAssign, Neg};

use num_traits::Float;

use crate::prelude::*;

pub trait BackwardTrait<F> {
    fn backward(&self, storage: &mut ArrayStorage<F>) -> Result<(), PzeudoErr>;
}

impl<F> BackwardTrait<F> for RecordLabel
where
    F: AddAssign + Copy + Neg<Output = F> + Float,
{
    fn backward(&self, storage: &mut ArrayStorage<F>) -> Result<(), PzeudoErr> {
        match self {
            Self::Add(lhs, rhs, grad) => {
                add_backward(*grad, lhs.1, rhs.1, storage)?;
            }
            Self::Sub(lhs, rhs, grad) => {
                sub_backward(*grad, lhs.1, rhs.1, storage)?;
            }
            Self::Div(lhs, rhs, grad) => {
                div_backward(*grad, lhs.0, rhs.0, lhs.1, rhs.1, storage)?;
            }
            Self::Mul(lhs, rhs, grad) => {
                mul_backward(*grad, lhs.0, rhs.0, lhs.1, rhs.1, storage)?;
            }
        }
        Ok(())
    }
}
