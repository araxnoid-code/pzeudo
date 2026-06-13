use std::sync::{Arc, RwLock};

use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

use crate::{BackLabel, NDArrayF64, TensorInner};

pub struct Tensor<'a> {
    inner: Arc<TensorInner<'a>>,
}

impl<'a> Tensor<'a> {
    pub fn new(
        array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>,
        grad: Option<NDArrayF64<'a>>,
        back_label: Option<BackLabel<'a>>,
    ) -> Tensor<'a> {
        let shape = array.shape().to_vec();
        Self {
            inner: Arc::new(TensorInner::new(
                NDArrayF64::OwnRepr(RwLock::new(array)),
                grad,
                shape,
                back_label,
            )),
        }
    }

    pub fn add(&'a self, rhs: &Self) -> Self {
        let mut inner = self.inner.add(&rhs.inner);
        inner.back_label = Some(BackLabel::Add(
            Arc::clone(&self.inner),
            Arc::clone(&rhs.inner),
        ));
        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn sub(&'a self, rhs: &Self) -> Self {
        let mut inner = self.inner.sub(&rhs.inner);
        inner.back_label = Some(BackLabel::Add(
            Arc::clone(&self.inner),
            Arc::clone(&rhs.inner),
        ));
        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn mul(&'a self, rhs: &Self) -> Self {
        let mut inner = self.inner.mul(&rhs.inner);
        inner.back_label = Some(BackLabel::Add(
            Arc::clone(&self.inner),
            Arc::clone(&rhs.inner),
        ));
        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn div(&'a self, rhs: &Self) -> Self {
        let mut inner = self.inner.div(&rhs.inner);
        inner.back_label = Some(BackLabel::Add(
            Arc::clone(&self.inner),
            Arc::clone(&rhs.inner),
        ));
        Tensor {
            inner: Arc::new(inner),
        }
    }
}
