use std::ops::Add;

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::NDArrayDataType;

pub struct F64Base {
    array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
}

impl F64Base {
    pub fn new(array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>) -> F64Base {
        Self { array }
    }
}

impl NDArrayDataType for F64Base {
    fn add(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).add(&rhs.array),
        }
    }
}
