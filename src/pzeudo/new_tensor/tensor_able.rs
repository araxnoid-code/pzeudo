use std::ops::AddAssign;

use ndarray::{ArrayD, ArrayView, ArrayViewD, CowArray, Dim, IxDynImpl};

use crate::OpsAble;

pub trait TensorAble {
    type OpsType<'a>: OpsAble
    where
        Self: 'a;

    // ops able
    fn to_ops_able(&self) -> Self::OpsType<'_>;

    // getter
    fn get_shape(&self) -> &[usize];
}

pub trait TensorMutAble: TensorAble {
    fn tensor_add_assign(&mut self, rhs: <Self as TensorAble>::OpsType<'_>);
}

impl TensorAble for ArrayD<f32> {
    type OpsType<'a> = ArrayViewD<'a, f32>;
    fn to_ops_able(&self) -> ArrayViewD<'_, f32> {
        self.view()
    }

    fn get_shape(&self) -> &[usize] {
        self.shape()
    }
}

impl TensorMutAble for ArrayD<f32> {
    fn tensor_add_assign(&mut self, rhs: ArrayViewD<'_, f32>) {
        self.add_assign(&rhs);
    }
}

impl TensorAble for ArrayViewD<'_, f32> {
    type OpsType<'a>
        = ArrayViewD<'a, f32>
    where
        Self: 'a;
    fn to_ops_able(&self) -> ArrayViewD<'_, f32> {
        self.view()
    }

    fn get_shape(&self) -> &[usize] {
        self.shape()
    }
}

// impl<'a> TensorAble for CowArray<'a, f32, Dim<IxDynImpl>> {
//     fn to_ops_able(&self) -> impl OpsAble {
//         self.view()
//     }

//     fn get_shape(&self) -> &[usize] {
//         self.shape()
//     }
// }
