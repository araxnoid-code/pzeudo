use std::marker::PhantomData;

pub struct ArrayRef<'a, F, T> {
    pub(crate) data: &'a [F],
    pub(crate) offset: usize,
    pub(crate) stride: &'a [usize],
    pub(crate) shape: &'a [usize],
    pub(crate) _array_type: PhantomData<T>,
}
