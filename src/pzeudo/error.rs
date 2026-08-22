use rand_distr::{BernoulliError, NormalError};

#[derive(Debug)]
pub enum PzeudoErr {
    // Tensor
    TensorErr(String),
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
    BernoulliErr(BernoulliError),
    SerdeJsonErr(serde_json::Error),
    IOErr(std::io::Error),
}
