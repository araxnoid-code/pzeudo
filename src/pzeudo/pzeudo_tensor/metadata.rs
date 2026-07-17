#[derive(Clone)]
pub struct TensorMetadata {
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
}
