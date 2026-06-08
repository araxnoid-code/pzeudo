use std::ops::{Add, Deref, Div, Mul, Sub};

use ndarray::{ArrayBase, ArrayD, ArrayRef, Dim, IxDynImpl, OwnedRepr, linalg::Dot};

use crate::PzeudoDataType;

pub struct F64Base {
    array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
    gradient: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
}

impl F64Base {
    pub fn new(array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>) -> F64Base {
        Self {
            gradient: ArrayD::zeros(array.shape()),
            array,
        }
    }
}

impl Deref for F64Base {
    type Target = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>;
    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl PzeudoDataType for F64Base {
    //
    type DataType = f64;

    //
    fn add(&self, rhs: &Self) -> Self {
        let array = (&self.array).add(&rhs.array);
        let gradient = ArrayD::zeros(array.shape());
        F64Base { array, gradient }
    }

    //
    fn sub(&self, rhs: &Self) -> Self {
        let array = (&self.array).sub(&rhs.array);
        let gradient = ArrayD::zeros(array.shape());
        F64Base { array, gradient }
    }

    //
    fn mul(&self, rhs: &Self) -> Self {
        let array = (&self.array).mul(&rhs.array);
        let gradient = ArrayD::zeros(array.shape());
        F64Base { array, gradient }
    }

    //
    fn div(&self, rhs: &Self) -> Self {
        let array = (&self.array).div(&rhs.array);
        let gradient = ArrayD::zeros(array.shape());
        F64Base { array, gradient }
    }
}
