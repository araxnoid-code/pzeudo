use crate::StorageType;

pub enum RecordLabel {
    // Arithmetic
    Add(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // Current Grad
    ),
    Div(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // Current Grad
    ),
    Mul(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // Current Grad
    ),
    Sub(
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (StorageType, Option<StorageType>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<StorageType>,                                    // Current Grad
    ),
    // Matmul
    Matmul2dF32(
        (StorageType, Option<StorageType>),
        (StorageType, Option<StorageType>),
        Option<StorageType>,
    ),
    Matmul2dF64(
        (StorageType, Option<StorageType>),
        (StorageType, Option<StorageType>),
        Option<StorageType>,
    ),
}
