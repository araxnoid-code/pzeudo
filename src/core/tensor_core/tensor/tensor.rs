use crate::tensor_core::GraphType;

pub struct Tensor {
    pub(crate) data: f64,
    pub(crate) graph_type: GraphType,
}
