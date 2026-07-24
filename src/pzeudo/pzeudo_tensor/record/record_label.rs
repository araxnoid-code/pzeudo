use std::fmt::Debug;

use crate::StorageType;

pub enum RecordLabel {
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
    // LOSS
    LossMse(StorageType, Option<StorageType>, Option<StorageType>), // (Output, Prediction grad, Own Grad)
}

impl Debug for RecordLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add(_, _, _) => f.write_str("add record"),
            Self::Sub(_, _, _) => f.write_str("sub record"),
            Self::Mul(_, _, _) => f.write_str("mul record"),
            Self::Div(_, _, _) => f.write_str("div record"),
            Self::LossMse(_, _, _) => f.write_str("Loss Mse record"),
            Self::Matmul2dF32(_, _, _) => f.write_str("Matmul 2d f32 record"),
            Self::Matmul2dF64(_, _, _) => f.write_str("Matmul 2d f64 record"),
        }
    }
}
