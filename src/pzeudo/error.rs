use rand_distr::NormalError;

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

    // Layer
    LayerErr(String),

    // Optim
    OptimErr(String),

    // Module
    ModuleErr(String),

    // DependenciesErr
    RandDistrNormalErr(NormalError),
    SerdeJsonErr(serde_json::Error),
    IOErr(std::io::Error),
}
