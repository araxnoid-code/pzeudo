use std::marker::PhantomData;

pub struct ArrayRefMut<'a, F, T> {
    pub(crate) data: &'a mut [F],
    pub(crate) offset: usize,
    pub(crate) stride: &'a [usize],
    pub(crate) shape: &'a [usize],
    pub(crate) _array_type: PhantomData<T>,
}
