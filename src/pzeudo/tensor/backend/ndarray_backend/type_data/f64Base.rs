use std::ops::{Add, Div, Mul, Sub};

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::{NDArrayDataType, PzeudoDataType};

pub struct F64Base {
    array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
}

impl F64Base {
    pub fn new(array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>) -> F64Base {
        Self { array }
    }
}

impl NDArrayDataType for F64Base {
    type ScalarType = PzeudoDataType;

    // desc
    fn get_array(&self) -> &Self {
        self
    }

    fn get_shape(&self) -> &[usize] {
        self.array.shape()
    }

    // initial
    fn from_scalar(scalar: Self::ScalarType) -> Self {
        match scalar {
            PzeudoDataType::F64(scalar) => Self {
                array: ArrayD::<f64>::from_elem(&[1_usize][..], scalar),
            },
            PzeudoDataType::I32(scalar) => Self {
                array: ArrayD::<f64>::from_elem(&[1_usize][..], scalar as f64),
            },
        }
    }

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

    // setter
    fn add_to(&mut self, lhs: &Self) {
        self.array = &self.array + &lhs.array;
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

    // scalar element wise
    // fn scalar_mul(&self, rhs: Self::ScalarType) -> Self {
    //     Self {
    //         array: &self.array * rhs,
    //     }
    // }
}
