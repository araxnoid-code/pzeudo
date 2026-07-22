use crate::{Array, TensorMetadata};

pub enum ElementType<F> {
    UpdateableTensor(usize),
    Contiguous(Array<F>, ContiguousType),
    View(usize, TensorMetadata),
}

#[derive(Debug)]
pub enum ContiguousType {
    Arr,
    Grad,
}
