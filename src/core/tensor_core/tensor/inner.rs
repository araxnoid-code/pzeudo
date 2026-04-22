use crate::tensor_core::GraphType;

pub struct InnerTensor {
    pub(crate) data: f64,
    pub(crate) graph_type: GraphType,
}

impl InnerTensor {
    pub(crate) fn add(&self, other: &InnerTensor) -> InnerTensor {
        InnerTensor {
            data: self.data + other.data,
            graph_type: GraphType::Schedule,
        }
    }

    pub(crate) fn sub(&self, other: &InnerTensor) -> InnerTensor {
        InnerTensor {
            data: self.data - other.data,
            graph_type: GraphType::Schedule,
        }
    }

    pub(crate) fn mul(&self, other: &InnerTensor) -> InnerTensor {
        InnerTensor {
            data: self.data * other.data,
            graph_type: GraphType::Schedule,
        }
    }

    pub(crate) fn div(&self, other: &InnerTensor) -> InnerTensor {
        InnerTensor {
            data: self.data / other.data,
            graph_type: GraphType::Schedule,
        }
    }
}
