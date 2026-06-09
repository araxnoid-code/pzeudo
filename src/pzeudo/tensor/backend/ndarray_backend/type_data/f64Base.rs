use std::ops::{Add, Div, Mul, Sub};

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
    // desc
    fn get_shape(&self) -> &[usize] {
        self.array.shape()
    }

    // initial
    fn ones(shape: &[usize]) -> Self {
        Self {
            array: ArrayD::<f64>::ones(shape),
        }
    }

    fn zeros(shape: &[usize]) -> Self {
        Self {
            array: ArrayD::<f64>::zeros(shape),
        }
    }

    // element wise
    fn add(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).add(&rhs.array),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).sub(&rhs.array),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).mul(&rhs.array),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).div(&rhs.array),
        }
    }
}
