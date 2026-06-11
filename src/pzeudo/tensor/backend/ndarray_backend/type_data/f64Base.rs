use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::{NDArrayDataType, PzeudoDataType};

pub struct F64Base<'a> {
    array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
    _phantom: PhantomData<&'a i32>,
}

impl<'a> F64Base<'a> {
    pub fn new(array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>) -> F64Base<'a> {
        Self {
            array,
            _phantom: PhantomData::default(),
        }
    }
}

impl<'a> NDArrayDataType<'a> for F64Base<'a> {
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
                _phantom: PhantomData::default(),
            },
            PzeudoDataType::I32(scalar) => Self {
                array: ArrayD::<f64>::from_elem(&[1_usize][..], scalar as f64),
                _phantom: PhantomData::default(),
            },
        }
    }

    fn ones(shape: &[usize]) -> Self {
        Self {
            array: ArrayD::<f64>::ones(shape),
            _phantom: PhantomData::default(),
        }
    }

    fn zeros(shape: &[usize]) -> Self {
        Self {
            array: ArrayD::<f64>::zeros(shape),
            _phantom: PhantomData::default(),
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
            _phantom: PhantomData::default(),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).sub(&rhs.array),
            _phantom: PhantomData::default(),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).mul(&rhs.array),
            _phantom: PhantomData::default(),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        Self {
            array: (&self.array).div(&rhs.array),
            _phantom: PhantomData::default(),
        }
    }

    // scalar element wise
    // fn scalar_mul(&self, rhs: Self::ScalarType) -> Self {
    //     Self {
    //         array: &self.array * rhs,
    //     }
    // }
}
