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
    Index(String),

    // // Arith
    AddErr(String),
    AddAssignErr(String),
    SubErr(String),
    MulErr(String),
    DivErr(String),

    // // Matmul
    Matmul2DErr(String),
    MatmulNDErr(String),

    // // View
    SlicingErr(String),
    BroadcastErr(String),
    PermuteErr(String),

    // // Reduction
    SumAxisErr(String),
    AvgErr(String),

    // // DotProduct
    DotProductErr(String),

    // Storage
    StoragePushErr(String),
    StorageGetErr(String),
    StorageGetAsArrayRefErr(String),
    StorageRemoveErr(String),
}
