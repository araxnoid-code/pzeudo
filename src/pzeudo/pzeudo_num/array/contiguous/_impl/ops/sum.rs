use std::iter::Sum;

use crate::{Array, OpsSum};

impl<F> OpsSum<F> for Array<F>
where
    for<'a> F: Copy + Sum<&'a F>,
{
    fn sum(&self) -> Result<Array<F>, crate::PzeudoNumErr>
    where
        F: Copy + std::ops::AddAssign + num_traits::Zero,
    {
        let sum = self.data.iter().sum::<F>();
        Ok(Array {
            data: vec![sum],
            offset: 0,
            stride: vec![1],
            shape: vec![1],
        })
    }
}
