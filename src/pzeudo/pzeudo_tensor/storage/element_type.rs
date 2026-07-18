use crate::{Array, TensorMetadata};

pub enum ElementType<F> {
    Contiguous(Array<F>),
    View(usize, TensorMetadata),
}
