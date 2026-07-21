pub enum RecordLabel {
    // Arithmetic
    Add(
        (usize, Option<usize>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (usize, Option<usize>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<usize>,                              // Current Grad
    ),
    Div(
        (usize, Option<usize>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (usize, Option<usize>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<usize>,                              // Current Grad
    ),
    Mul(
        (usize, Option<usize>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (usize, Option<usize>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<usize>,                              // Current Grad
    ),
    Sub(
        (usize, Option<usize>, Option<Vec<usize>>), // Lhs(arr, Grad)
        (usize, Option<usize>, Option<Vec<usize>>), // Rhs(arr, grad)
        Option<usize>,                              // Current Grad
    ),
    // Matmul
    Matmul2dF32(
        (usize, Option<usize>),
        (usize, Option<usize>),
        Option<usize>,
    ),
    Matmul2dF64(
        (usize, Option<usize>),
        (usize, Option<usize>),
        Option<usize>,
    ),
}
