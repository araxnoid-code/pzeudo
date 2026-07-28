use std::fmt::{Debug, Display};

use crate::StorageType;

pub enum RecordLabel<F> {
    // Arithmetic
    Add(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // own Grad
    ),
    Div(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // own Grad
    ),
    Mul(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // own Grad
    ),
    Sub(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // own Grad
    ),
    // Matmul
    Matmul2dF32(
        (StorageType, Option<StorageType>), // Lhs(arr, grad)
        (StorageType, Option<StorageType>), // Rhs(arr, grad)
        Option<StorageType>,                // own grad
    ),
    Matmul2dF64(
        (StorageType, Option<StorageType>), // Lhs(arr, grad)
        (StorageType, Option<StorageType>), // Rhs(arr, grad)
        Option<StorageType>,                // own grad
    ),
    MatmulNdF32(
        (StorageType, Option<StorageType>), // Lhs(arr, grad)
        (StorageType, Option<StorageType>), // Rhs(arr, grad)
        Option<StorageType>,                // own grad
    ),
    MatmulNdF64(
        (StorageType, Option<StorageType>), // Lhs(arr, grad)
        (StorageType, Option<StorageType>), // Rhs(arr, grad)
        Option<StorageType>,                // own grad
    ),

    // Unary
    Log((StorageType, Option<StorageType>), F, Option<StorageType>),
    Ln((StorageType, Option<StorageType>), Option<StorageType>),
    Powi((StorageType, Option<StorageType>), i32, Option<StorageType>),
    Powf((StorageType, Option<StorageType>), F, Option<StorageType>),

    // LOSS
    LossMse(StorageType, Option<StorageType>, Option<StorageType>), // (Output, Prediction grad, Own Grad)
}

impl<F> Debug for RecordLabel<F>
where
    F: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add(_, _, _) => f.write_str("add record"),
            Self::Sub(_, _, _) => f.write_str("sub record"),
            Self::Mul(_, _, _) => f.write_str("mul record"),
            Self::Div(_, _, _) => f.write_str("div record"),
            Self::LossMse(_, _, _) => f.write_str("Loss Mse record"),
            Self::Matmul2dF32(_, _, _) => f.write_str("Matmul 2d f32 record"),
            Self::Matmul2dF64(_, _, _) => f.write_str("Matmul 2d f64 record"),
            Self::MatmulNdF32(_, _, _) => f.write_str("Matmul nd f32 record"),
            RecordLabel::MatmulNdF64(_, _, _) => f.write_str("Matmul nd f64 record"),
            Self::Log((_, _), base, _) => f.write_str(&format!("log base {base} record")),
            RecordLabel::Ln(_, _) => f.write_str(&format!("log natural record")),
            Self::Powi(_, i, _) => f.write_str(&format!("powi {i} record")),
            Self::Powf(_, float, _) => f.write_str(&format!("powf {float} record")),
        }
    }
}
