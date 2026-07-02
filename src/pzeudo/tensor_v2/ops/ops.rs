use ndarray::{ArrayD, ArrayViewD, Axis};

use crate::{ArrayAble, ArrayRef};

pub trait OpsAble {
    type Array: ArrayAble;

    // Arith
    fn _add(&self, rhs: &Self) -> Self::Array;

    // Sum
    fn _sum_axis(&self, axis: usize) -> Self::Array;

    // shape
    fn _to_shape<ArrayR>(&self, shape: &[usize])
    where
        ArrayR: ArrayRef,
    {
    }
}

impl OpsAble for ArrayViewD<'_, f32> {
    type Array = ArrayD<f32>;

    fn _add(&self, rhs: &Self) -> Self::Array {
        self + rhs
    }

    // sum
    fn _sum_axis(&self, axis: usize) -> Self::Array {
        self.sum_axis(Axis(axis))
    }

    fn _to_shape<ArrayR>(&self, shape: &[usize])
    where
        ArrayR: ArrayRef,
    {
    }
    // fn _to_shape(&self, shape: &[usize]) {
    //     let reshape = self.to_shape(shape).unwrap();
    // }
}
