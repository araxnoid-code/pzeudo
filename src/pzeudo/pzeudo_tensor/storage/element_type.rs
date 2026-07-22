use std::panic;

use crate::{Array, TensorMetadata};

#[derive(Debug)]
pub enum ContiguousType {
    Arr,
    Grad,
}

pub enum ElementType<F> {
    PermanentTensor(usize),
    Contiguous(Array<F>, ContiguousType),
    View(usize, TensorMetadata),
}

impl<F> Clone for ElementType<F> {
    fn clone(&self) -> Self {
        match self {
            Self::PermanentTensor(idx) => Self::PermanentTensor(*idx),
            _ => panic!(
                "PZEUDO PANIC, CLONE ON ContiguousType ONLY POSSIBLE ON ContiguousType::PermanentTensor(usize)"
            ),
        }
    }
}
