use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};

use crate::OpsAble;

pub trait TensorAble {
    fn to_ops_able(&self) -> impl OpsAble;
    fn get_shape(&self) -> &[usize];
}

impl TensorAble for ArrayD<f32> {
    fn to_ops_able(&self) -> impl OpsAble {
        self.view()
    }

    fn get_shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<'a> TensorAble for ArrayViewD<'a, f32> {
    fn to_ops_able(&self) -> impl OpsAble {
        self.view()
    }

    fn get_shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<'a> TensorAble for CowArray<'a, f32, Dim<IxDynImpl>> {
    fn to_ops_able(&self) -> impl OpsAble {
        self.view()
    }

    fn get_shape(&self) -> &[usize] {
        self.shape()
    }
}
