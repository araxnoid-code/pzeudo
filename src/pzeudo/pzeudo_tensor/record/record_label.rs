pub enum RecordLabel {
    // Arithmetic
    Add(
        (usize, Option<usize>), // Lhs(arr, Grad)
        (usize, Option<usize>), // Rhs(arr, grad)
        Option<usize>,          // Current Grad
    ),
    Div(
        (usize, Option<usize>), // Lhs(arr, Grad)
        (usize, Option<usize>), // Rhs(arr, grad)
        Option<usize>,          // Current Grad
    ),
    Mul(
        (usize, Option<usize>), // Lhs(arr, Grad)
        (usize, Option<usize>), // Rhs(arr, grad)
        Option<usize>,          // Current Grad
    ),
    Sub(
        (usize, Option<usize>), // Lhs(arr, Grad)
        (usize, Option<usize>), // Rhs(arr, grad)
        Option<usize>,          // Current Grad
    ),
}
