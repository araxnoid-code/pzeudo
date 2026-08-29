use crate::{ConcatGradStatus, StorageType};

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

    // Reduction
    Sum(Option<StorageType>, Option<StorageType>), // Array Gradient, Gradient
    SumAxis(Option<StorageType>, Vec<usize>, bool, Option<StorageType>), // ArrayGradient, Reduction Axis, Gradient
    Avg(Option<StorageType>, Option<StorageType>),                       // Array Gradient, Gradient
    AvgAxis(Option<StorageType>, Vec<usize>, bool, Option<StorageType>), // ArrayGradient, Reduction Axis, Gradient

    // Flatten
    Flatten(Option<StorageType>, Vec<usize>, Option<StorageType>),

    // Concat
    Concat(Vec<ConcatGradStatus>, usize, Option<StorageType>),

    // Unary
    Log((StorageType, Option<StorageType>), F, Option<StorageType>),
    Ln((StorageType, Option<StorageType>), Option<StorageType>),
    Powi((StorageType, Option<StorageType>), i32, Option<StorageType>),
    Powf((StorageType, Option<StorageType>), F, Option<StorageType>),
    Sqrt((StorageType, Option<StorageType>), Option<StorageType>),
    Exp((StorageType, Option<StorageType>), Option<StorageType>),
    Sin((StorageType, Option<StorageType>), Option<StorageType>),
    Cos((StorageType, Option<StorageType>), Option<StorageType>),
    Tan((StorageType, Option<StorageType>), Option<StorageType>),

    // Activation
    Softplus(StorageType, Option<StorageType>, Option<StorageType>),
    Relu(StorageType, Option<StorageType>, Option<StorageType>),
    Sigmoid(StorageType, Option<StorageType>, Option<StorageType>),
    Tanh(StorageType, Option<StorageType>, Option<StorageType>),

    // LOSS
    LossMse(
        StorageType,
        StorageType,
        Option<StorageType>,
        Option<StorageType>,
    ), // (actual_idx, prediction_idx, Prediction grad, Own Grad)
    LossMae(
        StorageType,
        StorageType,
        Option<StorageType>,
        Option<StorageType>,
    ), // (actual_idx, prediction_idx, Prediction grad, Own Grad)
    CrossEntropyLoss(
        StorageType,
        StorageType,
        Option<StorageType>,
        Option<StorageType>,
    ), // (actual, prediction, prediction_grad, grad),

    // LAYER
    Dropout(Vec<u8>, F, Option<StorageType>, Option<StorageType>), // (Mask, p - 1, Array's Grad, Grad)
    LayerNorm(
        StorageType,                                // Array
        Option<StorageType>,                        // Array's Grad
        Vec<F>,                                     // variance
        Option<(StorageType, Option<StorageType>)>, // (gamma, gamma_grad)
        Option<(StorageType, Option<StorageType>)>, // (beta, beta_grad)
        Option<StorageType>,                        // Grad
    ),
}

// impl<F> Debug for RecordLabel<F>
// where
//     F: Display,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             RecordLabel::Reduction(_) =>
//             Self::Add(_, _, _) => f.write_str("add record"),
//             Self::Sub(_, _, _) => f.write_str("sub record"),
//             Self::Mul(_, _, _) => f.write_str("mul record"),
//             Self::Div(_, _, _) => f.write_str("div record"),
//             Self::LossMse(_, _, _, _) => f.write_str("Loss Mse record"),
//             Self::LossMae(_, _, _, _) => f.write_str("Loss Mae record"),
//             Self::CrossEntropyLoss(_, _, _, _) => f.write_str("Cross Entropy Loss record"),
//             Self::Matmul2dF32(_, _, _) => f.write_str("Matmul 2d f32 record"),
//             Self::Matmul2dF64(_, _, _) => f.write_str("Matmul 2d f64 record"),
//             Self::MatmulNdF32(_, _, _) => f.write_str("Matmul nd f32 record"),
//             RecordLabel::MatmulNdF64(_, _, _) => f.write_str("Matmul nd f64 record"),
//             Self::Log((_, _), base, _) => f.write_str(&format!("log base {base} record")),
//             RecordLabel::Ln(_, _) => f.write_str(&format!("log natural record")),
//             Self::Powi(_, i, _) => f.write_str(&format!("powi {i} record")),
//             Self::Powf(_, float, _) => f.write_str(&format!("powf {float} record")),
//             Self::Sqrt(_, _) => f.write_str(&format!("sqrt record")),
//             RecordLabel::Softplus(_, _, _) => f.write_str(&format!("softplus record")),
//             Self::Exp(_, _) => f.write_str(&format!("exp record")),
//             Self::Relu(_, _, _) => f.write_str(&format!("relu record")),
//             RecordLabel::Sigmoid(_, _, _) => f.write_str(&format!("sigmoid record")),
//             Self::Tanh(_, _, _) => f.write_str(&format!("tanh record")),
//         }
//     }
// }
