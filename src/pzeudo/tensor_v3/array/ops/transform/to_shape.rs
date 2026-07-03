use num_traits::Float;

use crate::ArrayTrait;

pub trait ArrayToShape<F>
where
    F: Float,
{
    type Output<'arr>
    where
        Self: 'arr;

    fn _to_shape(&self, new_shape: &[usize]) -> Self::Output<'_>;
}
