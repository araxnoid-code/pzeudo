pub enum RecordLabel {
    Add(
        (usize, Option<usize>), // Lhs(arr, Grad)
        (usize, Option<usize>), // Rhs(arr, grad)
        Option<usize>,          // Current Grad
    ),
}
