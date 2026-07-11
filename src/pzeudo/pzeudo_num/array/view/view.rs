pub struct ArrayView<'v, F> {
    data: &'v Vec<F>,
    offset: usize,
    stride: Vec<usize>,
    shape: Vec<usize>,
}
