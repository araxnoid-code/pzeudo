#[derive(Debug)]
pub enum PzeudoErr {
    // Tensor Tools
    TensorToolsErr(String),

    // Array
    ArrayErr(String),

    // GradStats
    ReqGradErr(String),

    // Ops
    OpsErr(String),

    // Storage
    StorageErr(String),
    StorageNoGradErr(String),
    StorageTimeErr(String),

    // Backward
    BackwardErr(String),

    // Loss
    LossErr(String),
}
