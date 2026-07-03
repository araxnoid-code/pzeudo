use ndarray::{ArrayD, ArrayViewD, Axis, CowArray, Dim, IxDynImpl};

use crate::ArrayAble;

pub trait OpsAble {}

impl OpsAble for ArrayViewD<'_, f32> {}

// Arith
pub trait AddOps {
    type Out: ArrayAble;

    fn _add(&self, rhs: &Self) -> Self::Out;
}

// Sum
pub trait SumOps {
    type Out: ArrayAble;

    fn _sum_axis(&self, axis: usize) -> Self::Out;
}

// ToShape
pub trait ToShape: ArrayAble {
    type Out<'out>: ArrayAble<Ops<'out> = Self::Ops<'out>>
    where
        Self: 'out;

    fn _to_shape(&self, new_shape: &[usize]) -> Self::Out<'_>;
}

// impl
impl AddOps for ArrayViewD<'_, f32> {
    type Out = ArrayD<f32>;

    fn _add(&self, rhs: &Self) -> Self::Out {
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
    type Out<'out>
        = CowArray<'out, f32, Dim<IxDynImpl>>
    where
        Self: 'out;

    fn _to_shape(&self, new_shape: &[usize]) -> Self::Out<'_> {
        self.to_shape(new_shape).unwrap()
    }
}
