use ndarray::{ArrayD, CowArray, Dim, IxDynImpl};
use num_traits::Float;

use crate::{Array, ArrayToShape, ArrayTrait};

impl<F> ArrayTrait for ArrayD<F> where F: Float {}

impl<F> ArrayToShape<F> for Array<ArrayD<F>>
where
    F: Clone + Float + 'static,
{
    type Output<'arr>
        = CowArray<'arr, F, Dim<IxDynImpl>>
    where
        Self: 'arr;
    fn _to_shape(&self, new_shape: &[usize]) -> Self::Output<'_> {
        self.array.to_shape(new_shape).unwrap()
    }
}
