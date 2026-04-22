use cahotic::OutputTrait;

use crate::tensor_core::InnerTensor;

pub struct ExecuteOutput(pub Result<InnerTensor, ()>);
impl OutputTrait for ExecuteOutput {}
