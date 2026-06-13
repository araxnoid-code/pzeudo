use std::sync::RwLock;

use ndarray::ArrayD;

use crate::{BackLabel, NDArrayF64};

pub struct TensorInner<'a> {
    pub(crate) array: NDArrayF64<'a>,
    pub(crate) grad: Option<NDArrayF64<'a>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) back_label: Option<BackLabel<'a>>,
}

impl<'a> TensorInner<'a> {
    pub fn new(
        array: NDArrayF64<'a>,
        grad: Option<NDArrayF64<'a>>,
        shape: Vec<usize>,
        back_label: Option<BackLabel<'a>>,
    ) -> TensorInner<'a> {
        Self {
            array,
            grad: grad.map(|grad| grad),
            shape,
            back_label,
        }
    }

    pub fn add(&'a self, rhs: &Self) -> Self {
        Self {
            array: self.array.add(&rhs.array),
            grad: Some(NDArrayF64::OwnRepr(RwLock::new(ArrayD::<f64>::zeros(
                &self.shape[..],
            )))),
            back_label: None,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&'a self, rhs: &Self) -> Self {
        Self {
            array: self.array.sub(&rhs.array),
            grad: Some(NDArrayF64::OwnRepr(RwLock::new(ArrayD::<f64>::zeros(
                &self.shape[..],
            )))),
            back_label: None,
            shape: self.shape.clone(),
        }
    }

    pub fn mul(&'a self, rhs: &Self) -> Self {
        Self {
            array: self.array.mul(&rhs.array),
            grad: Some(NDArrayF64::OwnRepr(RwLock::new(ArrayD::<f64>::zeros(
                &self.shape[..],
            )))),
            back_label: None,
            shape: self.shape.clone(),
        }
    }

    pub fn div(&'a self, rhs: &Self) -> Self {
        Self {
            array: self.array.div(&rhs.array),
            grad: Some(NDArrayF64::OwnRepr(RwLock::new(ArrayD::<f64>::zeros(
                &self.shape[..],
            )))),
            back_label: None,
            shape: self.shape.clone(),
        }
    }
}
