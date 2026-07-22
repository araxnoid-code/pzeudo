#[derive(Debug)]
pub enum PzeudoErr {
    // tools
    ShapeToStride(String),

    // Array
    ArrayNewErr(String),
    ToStringErr(String),

    // Ops
    // //  index
    LinearIndexErr(String),
    MutLinearIndexErr(String),
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

    // Storage
    StoragePushErr(String),
    StorageGetErr(String),
    StorageGetAsArrayRefErr(String),
    StorageGetAsArrayRefMutErr(String),
    StorageRemoveErr(String),

    // backward
    BackwardErr(String),

    // module
    EpochErr(String),
}
