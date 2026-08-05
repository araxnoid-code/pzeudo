#[derive(Debug)]
pub enum PzeudoErr {
    // tools
    TensorToolsErr(String),

    // Array
    ArrayNewErr(String),
    ToStringErr(String),

    // Tensor
    ReqGradErr(String),

    // Ops
    // //  index
    LinearIndexErr(String),
    LinearIndexMutErr(String),
    Index(String),

    // // Arith
    AddErr(String),
    SubErr(String),
    MulErr(String),
    DivErr(String),

    // // Assign
    AssignErr(String),
    AddAssignErr(String),
    SubAssignErr(String),
    MulAssignErr(String),
    DivAssignErr(String),

    // // Matmul
    Matmul2DErr(String),
    MatmulNDErr(String),

    // // View
    SlicingErr(String),
    BroadcastErr(String),
    PermuteErr(String),
    ToShapeErr(String),

    // // Reduction
    SumAxisErr(String),
    AvgErr(String),

    // // DotProduct
    DotProductErr(String),

    // UNARY
    Log10Err(String),
    PowiBackwardErr(String),

    // METADATA
    CastingStorageTypeToView(String),

    // Storage
    StorageErr(String),
    StorageNoGradErr(String),
    StorageTimeErr(String),

    // backward
    BackwardErr(String),

    // module
    EpochErr(String),

    // Method
    LinearForward(String),

    // Loss
    MseErr(String),
    MseBackwardErr(String),
    MaeErr(String),
    MaeBackwardErr(String),
    CrossEntropyLossErr(String),
    CrossEntropyLossBackwardErr(String),
}
