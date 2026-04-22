use std::{rc::Rc, sync::Arc};

use crate::tensor_core::{GraphType, tensor::inner::InnerTensor};

pub struct Tensor {
    pub(crate) inner: Arc<InnerTensor>,
}

impl Tensor {
    pub(crate) fn init(data: f64, graph_type: GraphType) -> Tensor {
        Tensor {
            inner: Arc::new(InnerTensor { data, graph_type }),
        }
    }

    pub(crate) fn add(&self, other: &Tensor) -> Tensor {
        Tensor::init(self.inner.data + other.inner.data, GraphType::Schedule)
    }

    pub(crate) fn sub(&self, other: &Tensor) -> Tensor {
        Tensor::init(self.inner.data - other.inner.data, GraphType::Schedule)
    }

    pub(crate) fn mul(&self, other: &Tensor) -> Tensor {
        Tensor::init(self.inner.data * other.inner.data, GraphType::Schedule)
    }

    pub(crate) fn div(&self, other: &Tensor) -> Tensor {
        Tensor::init(self.inner.data / other.inner.data, GraphType::Schedule)
    }
}
