use std::{
    borrow::{self, Cow},
    cell::RefCell,
    ops::{Add, AddAssign},
    rc::Rc,
};

use ndarray::{ArrayD, ArrayRefD, ArrayViewD, CowArray, Dim, IxDynImpl, array, linalg::Dot};
use pzeudo::{Array, ArrayToShape, Tensor};

fn main() {
    let data = array![[1., 2., 3.]].into_dyn();
    let view: ArrayViewD<f32> = data.view();

    // operate(view);
}

// impl ToOps for ArrayViewD<'_, f32> {
//     type OpsType<'a>
//         = ArrayViewD<'a, f32>
//     where
//         Self: 'a;

//     fn to_ops(&self) -> Self::OpsType<'_> {
//         self.view()
//     }
// }

// impl ToBorrow for ArrayViewD<'_, f32> {
//     type BorrowType<'a>
//         = CowArray<'a, f32, Dim<IxDynImpl>>
//     where
//         Self: 'a;

//     fn to_borrow(&self) -> Self::BorrowType<'_> {
//         self.to_shape(vec![10]).unwrap()
//     }
// }

// impl ToOps for CowArray<'_, f32, Dim<IxDynImpl>> {
//     type OpsType<'a>
//         = ArrayViewD<'a, f32>
//     where
//         Self: 'a;

//     fn to_ops(&self) -> Self::OpsType<'_> {
//         self.view()
//     }
// }
