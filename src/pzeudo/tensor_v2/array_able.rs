use std::ops::AddAssign;

use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};

use crate::OpsAble;

pub trait ArrayAble {
    type Ops<'ops>: OpsAble
    where
        Self: 'ops;

    // into
    fn into_ops(&self) -> Self::Ops<'_>;

    // getter
    fn shape(&self) -> &[usize];
}

pub trait ArrayMutAble: ArrayAble {
    // setter
    fn _add_assign(&mut self, assign: Self::Ops<'_>);
}

pub trait ArrayRef {}
//

impl ArrayAble for ArrayD<f32> {
    type Ops<'ops>
        = ArrayViewD<'ops, f32>
    where
        Self: 'ops;

    // into
    fn into_ops(&self) -> Self::Ops<'_> {
        self.view()
    }

    // getter
    fn shape(&self) -> &[usize] {
        Self::shape(&self)
    }
}

impl ArrayMutAble for ArrayD<f32> {
    fn _add_assign(&mut self, assign: Self::Ops<'_>) {
        AddAssign::add_assign(self, &assign);
    }
}

impl ArrayAble for ArrayViewD<'_, f32> {
    type Ops<'ops>
        = ArrayViewD<'ops, f32>
    where
        Self: 'ops;

    // into
    fn into_ops(&self) -> Self::Ops<'_> {
        self.view()
    }

    // getter
    fn shape(&self) -> &[usize] {
        Self::shape(&self)
    }
}

impl ArrayAble for CowArray<'_, f32, Dim<IxDynImpl>> {
    type Ops<'ops>
        = ArrayViewD<'ops, f32>
    where
        Self: 'ops;

    // into
    fn into_ops(&self) -> Self::Ops<'_> {
        self.view()
    }

    // getter
    fn shape(&self) -> &[usize] {
        Self::shape(&self)
    }
}
