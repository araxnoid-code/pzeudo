use ndarray::{ArrayD, ArrayViewD};

use crate::OpsAble;

pub trait OwnAble {
    type OpsType<'a>: OpsAble
    where
        Self: 'a;

    fn to_ops_able(&self) -> Self::OpsType<'_>;
}
impl OwnAble for ArrayD<f32> {
    type OpsType<'a>
        = ArrayViewD<'a, f32>
    where
        Self: 'a;

    fn to_ops_able(&self) -> Self::OpsType<'_> {
        self.view()
    }
}

pub trait RefAble {
    type OwnType: OwnAble;
    fn to_own(self) -> Self::OwnType
    where
        Self: Sized;
}
impl RefAble for ArrayViewD<'_, f32> {
    type OwnType = ArrayD<f32>;
    fn to_own(self) -> Self::OwnType
    where
        Self: Sized,
    {
        self.to_owned()
    }
}
