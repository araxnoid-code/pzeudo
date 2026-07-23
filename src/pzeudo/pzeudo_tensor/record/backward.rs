use std::{
    format,
    iter::Sum,
    ops::{AddAssign, Neg},
};

use num_traits::Float;

use crate::prelude::*;

pub trait BackwardTrait<F> {
    fn backward(&self, storage: &mut ArrayStorage<F>) -> Result<(), PzeudoErr>;
}

impl<F> BackwardTrait<F> for RecordLabel
where
    ArrayStorage<F>: StorageF32F64,
    for<'a> F: AddAssign + Copy + Neg<Output = F> + Float + Sum<&'a F>,
    for<'a> ArrayRef<'a, F, Contiguous>: OpsBroadcast<F>,
    for<'a> ArrayRef<'a, F, View>: OpsBroadcast<F>,
{
    fn backward(&self, storage: &mut ArrayStorage<F>) -> Result<(), PzeudoErr> {
        match self {
            Self::Add(lhs, rhs, grad) => {
                add_backward(*grad, lhs.1, lhs.2.as_ref(), rhs.1, rhs.2.as_ref(), storage)?;
            }
            Self::Sub(lhs, rhs, grad) => {
                sub_backward(*grad, lhs.1, lhs.2.as_ref(), rhs.1, rhs.2.as_ref(), storage)?;
            }
            Self::Div(lhs, rhs, grad) => {
                div_backward(
                    *grad,
                    lhs.0,
                    rhs.0,
                    lhs.1,
                    lhs.2.as_ref(),
                    rhs.1,
                    rhs.2.as_ref(),
                    storage,
                )?;
            }
            Self::Mul(lhs, rhs, grad) => {
                mul_backward(
                    *grad,
                    lhs.0,
                    rhs.0,
                    lhs.1,
                    lhs.2.as_ref(),
                    rhs.1,
                    rhs.2.as_ref(),
                    storage,
                )?;
            }
            RecordLabel::Matmul2dF32(lhs, rhs, grad) => {
                matmul_2d_f32_backward(
                    lhs.0,
                    lhs.1,
                    rhs.0,
                    rhs.1,
                    *grad,
                    storage
                        .to_mut_f32()
                        .ok_or(PzeudoErr::BackwardErr(format!("BackwardTrait::backward. Cannot perform backward on matmul_2d of type f32 because storage is not of type f32")))?,
                )?;
            }
            RecordLabel::Matmul2dF64(lhs, rhs, grad) => {
                matmul_2d_f64_backward(
                    lhs.0,
                    lhs.1,
                    rhs.0,
                    rhs.1,
                    *grad,
                    storage
                        .to_mut_f64()
                        .ok_or(PzeudoErr::BackwardErr(format!("BackwardTrait::backward. Cannot perform backward on matmul_2d of type f64 because the storage is not of type f64")))?,
                )?;
            }
        }
        Ok(())
    }
}
