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

impl<F> BackwardTrait<F> for RecordLabel<F>
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

            RecordLabel::MatmulNdF32(lhs, rhs, grad) => {
                matmul_nd_f32_backward(
                    lhs.0,
                    lhs.1,
                    rhs.0,
                    rhs.1,
                    *grad,
                    storage
                        .to_mut_f32()
                        .ok_or(PzeudoErr::BackwardErr(format!("BackwardTrait::backward. Cannot perform backward on matmul_2d of type f64 because the storage is not of type f64")))?,
                )?;
            }

            RecordLabel::MatmulNdF64(lhs, rhs, grad) => {
                matmul_nd_f64_backward(
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

            RecordLabel::Log(lhs, base, grad) => {
                log_backward(lhs.0, lhs.1, *grad, *base, storage)?;
            }

            RecordLabel::Ln(lhs, grad) => {
                ln_backward(lhs.0, lhs.1, *grad, storage)?;
            }

            Self::Powf(lhs, f, grad) => {
                powf_backward(lhs.0, lhs.1, *f, *grad, storage)?;
            }

            Self::Powi(lhs, i, grad) => {
                powi_backward(lhs.0, lhs.1, *i, *grad, storage)?;
            }

            Self::Sqrt(lhs, grad) => {
                sqrt_backward(lhs.0, lhs.1, *grad, storage)?;
            }

            Self::Exp(lhs, grad) => {
                exp_backward(lhs.0, lhs.1, *grad, storage)?;
            }

            Self::Softplus(array_idx, array_grad_idx, grad) => {
                softplus_backward(*array_idx, *array_grad_idx, *grad, storage)?;
            }

            Self::Relu(array_idx, array_grad_idx, grad) => {
                relu_backward(*array_idx, *array_grad_idx, *grad, storage)?;
            }

            Self::Sigmoid(output_idx, array_grad_idx, grad) => {
                sigmoid_backward(*output_idx, *array_grad_idx, *grad, storage)?;
            }

            RecordLabel::Tanh(output_idx, array_grad_idx, grad) => {
                tanh_backward(*output_idx, *array_grad_idx, *grad, storage)?;
            }

            RecordLabel::Sum(array_grad, grad) => {
                sum_backward(*array_grad, *grad, storage)?;
            }

            RecordLabel::SumAxis(array_grad, axis, keep_dim, grad) => {
                sum_axis_backward(*array_grad, axis, *keep_dim, *grad, storage)?;
            }

            RecordLabel::Avg(array_grad, grad) => {
                avg_backward(*array_grad, *grad, storage)?;
            }

            RecordLabel::AvgAxis(array_grad, axis, keep_dim, grad) => {
                avg_axis_backward(*array_grad, axis, *keep_dim, *grad, storage)?;
            }

            RecordLabel::LossMse(actual_idx, prediction_idx, prediction_grad_idx, grad) => {
                mse_backward(
                    *grad,
                    *actual_idx,
                    *prediction_idx,
                    *prediction_grad_idx,
                    storage,
                )?;
            }

            Self::LossMae(actual_idx, prediction_idx, prediction_grad_idx, grad) => {
                mae_backward(
                    *grad,
                    *actual_idx,
                    *prediction_idx,
                    *prediction_grad_idx,
                    storage,
                )?;
            }

            Self::CrossEntropyLoss(actual_idx, prediction_idx, prediction_grad_idx, grad) => {
                cross_entropy_loss_backward(
                    *actual_idx,
                    *prediction_idx,
                    *prediction_grad_idx,
                    *grad,
                    storage,
                )?;
            }
        }
        Ok(())
    }
}
