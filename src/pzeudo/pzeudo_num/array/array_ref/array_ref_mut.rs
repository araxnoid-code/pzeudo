use crate::{ArrayAssignTrait, ArrayTrait, PzeudoErr};
use num_traits::One;
use std::marker::PhantomData;

pub struct ArrayRefMut<'a, F, T> {
    pub(crate) data: &'a mut [F],
    pub(crate) offset: usize,
    pub(crate) stride: &'a [usize],
    pub(crate) shape: &'a [usize],
    pub(crate) _array_type: PhantomData<T>,
}

impl<F, T> ArrayRefMut<'_, F, T> {
    pub fn to_ones(&mut self) -> Result<(), PzeudoErr>
    where
        for<'a> ArrayRefMut<'a, F, T>: ArrayTrait<F>,
        F: Copy + One,
    {
        let len = self.shape.iter().product::<usize>();
        for i in 0..len {
            *self.mut_linear_index(i)? = F::one();
        }
        Ok(())
    }
}
