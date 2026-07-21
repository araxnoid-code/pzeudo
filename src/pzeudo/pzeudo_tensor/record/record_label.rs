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
}
