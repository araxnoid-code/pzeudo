use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
    sync::{Arc, Mutex, RwLock},
};

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::{NDArrayDataType, PzeudoDataType};

pub struct F64Base {
    array: Arc<RwLock<ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>>>,
    shape: Vec<usize>,
}

impl F64Base {
    pub fn new(array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>) -> F64Base {
        Self {
            shape: array.shape().to_vec(),
            array: Arc::new(RwLock::new(array)),
        }
    }
}

impl Display for F64Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let read = self.array.read().unwrap();
        f.write_str(&format!("{}", read))
    }
}

impl NDArrayDataType for F64Base {
    type ArrType = F64Base;
    type ScalarType = PzeudoDataType;

    // desc
    fn get_array(&self) -> &Self::ArrType {
        &self
    }

    fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    // initial
    fn from_scalar(scalar: Self::ScalarType) -> Self {
        match scalar {
            PzeudoDataType::F64(scalar) => Self {
                array: Arc::new(RwLock::new(ArrayD::<f64>::from_elem(
                    &[1_usize][..],
                    scalar,
                ))),
                shape: vec![1],
            },
            PzeudoDataType::I32(scalar) => Self {
                array: Arc::new(RwLock::new(ArrayD::<f64>::from_elem(
                    &[1_usize][..],
                    scalar as f64,
                ))),
                shape: vec![1],
            },
        }
    }

    fn ones(shape: Vec<usize>) -> Self {
        Self {
            shape: shape.to_vec(),
            array: Arc::new(RwLock::new(ArrayD::<f64>::ones(shape))),
        }
    }

    fn zeros(shape: Vec<usize>) -> Self {
        Self {
            shape: shape.to_vec(),
            array: Arc::new(RwLock::new(ArrayD::<f64>::zeros(shape))),
        }
    }

    // setter
    fn add_to(&mut self, lhs: &Self) {
        let result = &*self.array.read().unwrap() + &*lhs.array.read().unwrap();
        self.array = Arc::new(RwLock::new(result));
    }

    // element wise
    fn add(&self, rhs: &Self) -> Self {
        Self {
            array: Arc::new(RwLock::new(
                (&*self.array.read().unwrap()).add(&*rhs.array.read().unwrap()),
            )),
            shape: self.shape.to_vec(),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            array: Arc::new(RwLock::new(
                (&*self.array.read().unwrap()).sub(&*rhs.array.read().unwrap()),
            )),
            shape: rhs.shape.to_vec(),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            array: Arc::new(RwLock::new(
                (&*self.array.read().unwrap()).mul(&*rhs.array.read().unwrap()),
            )),
            shape: rhs.shape.to_vec(),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        Self {
            array: Arc::new(RwLock::new(
                (&*self.array.read().unwrap()).div(&*rhs.array.read().unwrap()),
            )),
            shape: rhs.shape.to_vec(),
        }
    }

    // scalar element wise
    // fn scalar_mul(&self, rhs: Self::ScalarType) -> Self {
    //     Self {
    //         array: &self.array * rhs,
    //     }
    // }
}
