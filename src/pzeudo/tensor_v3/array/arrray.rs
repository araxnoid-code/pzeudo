use std::marker::PhantomData;

use num_traits::Float;

use crate::ArrayTrait;

pub struct Array<Arr>
where
    Arr: ArrayTrait,
{
    pub(crate) array: Arr,
}

impl<Arr> Array<Arr>
where
    Arr: ArrayTrait,
{
    pub fn new(array: Arr) -> Array<Arr> {
        Self { array }
    }
}
