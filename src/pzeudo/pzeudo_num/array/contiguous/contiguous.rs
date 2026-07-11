pub struct Array<F> {
    pub(crate) data: Vec<F>,
    pub(crate) offset: usize,
    pub(crate) stride: Vec<usize>,
    pub(crate) shape: Vec<usize>,
}
