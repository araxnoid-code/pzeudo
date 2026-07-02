use ndarray::{ArrayD, ArrayViewD, Axis, CowArray, Dim, IxDynImpl};

use crate::ArrayAble;

pub trait OpsAble {}

impl OpsAble for ArrayViewD<'_, f32> {}

// Arith
pub trait AddOps<'ops>: ArrayAble<'ops> {
    type Out<'out>: ArrayAble<'out>
    where
        Self: 'out;

    fn _add(&self, rhs: &Self) -> Self::Out<'_>;
}

// Sum
pub trait SumOps: ArrayAble {
    type Out<'out>: ArrayAble<'out>
    where
        Self: 'out;

    fn _sum_axis(&self, axis: usize) -> Self::Out<'_>;
}

// ToShape
pub trait ToShape: ArrayAble {
    type Out<'a>: ArrayAble
    where
        Self: 'a;

    fn _to_shape(&self, new_shape: &[usize]) -> Self::Out<'_>;
}

// impl
impl<'ops> AddOps<'ops> for ArrayViewD<'ops, f32> {
    type Out<'out>
        = ArrayD<f32>
    where
        Self: 'out;

    fn _add(&self, rhs: &Self) -> Self::Out<'_> {
        self + rhs
    }
}

impl SumOps for ArrayViewD<'_, f32> {
    type Out = ArrayD<f32>;
    fn _sum_axis(&self, axis: usize) -> Self::Out {
        self.sum_axis(Axis(axis))
    }
}

impl ToShape for ArrayViewD<'_, f32> {
    type Out<'a>
        = CowArray<'a, f32, Dim<IxDynImpl>>
    where
        Self: 'a;

    fn _to_shape(&self, new_shape: &[usize]) -> Self::Out<'_> {
        self.to_shape(new_shape).unwrap()
    }
}
