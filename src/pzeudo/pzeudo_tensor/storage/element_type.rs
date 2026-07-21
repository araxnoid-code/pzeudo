use crate::{Array, TensorMetadata};

pub enum ElementType<F> {
    UpdateableTensor(usize),
    Contiguous(Array<F>),
    View(usize, TensorMetadata),
}
