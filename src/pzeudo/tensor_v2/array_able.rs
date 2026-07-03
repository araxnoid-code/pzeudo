use std::ops::AddAssign;

use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};

use crate::OpsAble;

pub trait ArrayAble {
    type Ops<'a>: OpsAble
    where
        Self: 'a;

    // into
    fn to_ops(&self) -> Self::Ops<'_>;

    // getter
    fn shape(&self) -> &[usize];
}

pub trait ArrayMutAble: ArrayAble {
    // setter
    fn _add_assign<'a>(&mut self, assign: Self::Ops<'a>);
}

//

impl ArrayAble for ArrayD<f32> {
    type Ops<'a> = ArrayViewD<'a, f32>;

    // into
    fn to_ops(&self) -> Self::Ops<'_> {
        self.view()
    }

    // getter
    fn shape(&self) -> &[usize] {
        Self::shape(&self)
    }
}

// impl<'ops> ArrayMutAble<'ops> for ArrayD<f32> {
//     fn _add_assign(&mut self, assign: Self::Ops) {
//         AddAssign::add_assign(self, &assign);
//     }
// }

impl ArrayAble for ArrayViewD<'_, f32> {
    type Ops<'ops>
        = ArrayViewD<'ops, f32>
    where
        Self: 'ops;

    // into
    fn to_ops(&self) -> Self::Ops<'_> {
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
    fn to_ops(&self) -> Self::Ops<'_> {
        self.view()
    }

    // getter
    fn shape(&self) -> &[usize] {
        Self::shape(&self)
    }
}
