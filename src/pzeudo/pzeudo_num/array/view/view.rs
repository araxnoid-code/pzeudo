pub struct ArrayView<'v, F> {
    pub(crate) data: &'v [F],
    pub(crate) offset: usize,
    pub(crate) stride: Vec<usize>,
    pub(crate) shape: Vec<usize>,
}
