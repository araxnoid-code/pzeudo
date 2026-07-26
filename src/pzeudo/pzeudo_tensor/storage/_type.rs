use crate::{Array, PermanentTensor, TensorMetadata};

// 1
#[derive(Debug)]
pub enum ContiguousType {
    Arr,
    Grad,
}

// 16
#[derive(Clone, Copy, Debug)]
pub enum StorageType {
    Permanent(usize),
    Storage(usize),
}

pub enum GetElementOutput<'a, F> {
    Permanent(&'a PermanentTensor<F>),
    Storage(&'a ElementType<F>),
}

pub enum GetElementMutOutput<'a, F> {
    Permanent(&'a mut PermanentTensor<F>),
    Storage(&'a mut ElementType<F>),
}

pub enum ElementType<F> {
    Contiguous(Array<F>, ContiguousType),
    View(StorageType, TensorMetadata),
}
