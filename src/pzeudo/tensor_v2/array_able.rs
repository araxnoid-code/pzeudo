use std::ops::AddAssign;

use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};

use crate::OpsAble;

pub trait ArrayAble<'ops> {
    type Ops: OpsAble;

    // into
    fn into_ops(&'ops self) -> Self::Ops;

    // getter
    fn shape(&self) -> &[usize];
}

pub trait ArrayMutAble<'ops>: ArrayAble<'ops> {
    // setter
    fn _add_assign(&mut self, assign: Self::Ops);
}

//

impl<'ops> ArrayAble<'ops> for ArrayD<f32> {
    type Ops = ArrayViewD<'ops, f32>;

    // into
    fn into_ops(&'ops self) -> Self::Ops {
        self.view()
    }

    // getter
    fn shape(&self) -> &[usize] {
        Self::shape(&self)
    }
}

impl<'ops> ArrayMutAble<'ops> for ArrayD<f32> {
    fn _add_assign(&mut self, assign: Self::Ops) {
        AddAssign::add_assign(self, &assign);
    }
}

// impl ArrayAble for ArrayViewD<'_, f32> {
//     type Ops<'ops>
//         = ArrayViewD<'ops, f32>
//     where
//         Self: 'ops;

//     // into
//     fn into_ops(&self) -> Self::Ops<'_> {
//         self.view()
//     }

//     // getter
//     fn shape(&self) -> &[usize] {
//         Self::shape(&self)
//     }
// }

// impl ArrayAble for CowArray<'_, f32, Dim<IxDynImpl>> {
//     type Ops<'ops>
//         = ArrayViewD<'ops, f32>
//     where
//         Self: 'ops;

//     // into
//     fn into_ops(&self) -> Self::Ops<'_> {
//         self.view()
//     }

//     // getter
//     fn shape(&self) -> &[usize] {
//         Self::shape(&self)
//     }
// }

// impl ArrayRef for CowArray<'_, f32, Dim<IxDynImpl>> {}
