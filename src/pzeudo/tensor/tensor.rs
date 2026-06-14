use std::{
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock},
};

use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

use crate::{BackLabel, NDArrayF64, OwnTensorF64, TensorF64Inner};

pub struct Tensor<I, Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
    I: TensorF64Inner<Lhs = Lhs, Rhs = Rhs>,
{
    inner: Arc<I>,
    _lhs_phantom: PhantomData<Lhs>,
    _rhs_phantom: PhantomData<Rhs>,
}

impl<I, OwnLhs, OwnRhs> Tensor<I, OwnLhs, OwnRhs>
where
    OwnLhs: TensorF64Inner,
    OwnRhs: TensorF64Inner,
    I: TensorF64Inner<Lhs = OwnLhs, Rhs = OwnRhs>,
{
    pub fn add<'v, A, OtherLhs, OtherRhs>(
        &self,
        rhs: &Tensor<A, OtherLhs, OtherRhs>,
    ) -> Tensor<I, I, A>
    where
        OtherLhs: TensorF64Inner,
        OtherRhs: TensorF64Inner,
        A: TensorF64Inner<ViewArr<'v> = I::ViewArr<'v>, Lhs = OtherLhs, Rhs = OtherRhs>,
    {
        let mut inner = I::own_type_into(self.inner.add(&rhs.inner.view()));
        if let Some(back_label) = self.inner.get_back_label_mut() {
            let lock = back_label.lock().unwrap();
            // *back_label = BackLabel::Add(self.inner.clone(), rhs.inner.clone());
        }

        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn sub(&self, rhs: &Self) -> Tensor<I> {
        let mut inner = self.inner.sub(&rhs.inner.view());
        let back_label = inner.get_back_label_mut();
        // *back_label = Some(Mutex::new(BackLabel::Add(
        //     self.inner.clone(),
        //     rhs.inner.clone(),
        // )));

        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn mul(&self, rhs: &Self) -> Tensor<I> {
        let mut inner = self.inner.mul(&rhs.inner.view());
        let back_label = inner.get_back_label_mut();
        // *back_label = Some(Mutex::new(BackLabel::Add(
        //     self.inner.clone(),
        //     rhs.inner.clone(),
        // )));

        Tensor {
            inner: Arc::new(inner),
        }
    }

    pub fn div(&self, rhs: &Self) -> Tensor<I> {
        let mut inner = self.inner.div(&rhs.inner.view());
        let back_label = inner.get_back_label_mut();
        // *back_label = Some(Mutex::new(BackLabel::Add(
        //     self.inner.clone(),
        //     rhs.inner.clone(),
        // )));

        Tensor {
            inner: Arc::new(inner),
        }
    }
}
