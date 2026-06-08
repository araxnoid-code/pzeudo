use std::ops::{Add, Deref, Div, Mul, Sub};

use ndarray::{ArrayBase, ArrayRef, Dim, IxDynImpl, OwnedRepr, linalg::Dot};

use crate::PzeudoDataType;

pub struct F64Base {
    array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
}

impl F64Base {
    pub fn new(array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>) -> F64Base {
        Self { array }
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
        F64Base {
            array: (&self.array).add(&rhs.array),
        }
    }

    //
    fn sub(&self, rhs: &Self) -> Self {
        F64Base {
            array: (&self.array).sub(&rhs.array),
        }
    }

    //
    fn mul(&self, rhs: &Self) -> Self {
        F64Base {
            array: (&self.array).mul(&rhs.array),
        }
    }

    //
    fn div(&self, rhs: &Self) -> Self {
        F64Base {
            array: (&self.array).div(&rhs.array),
        }
    }
}
