#[derive(Debug)]
pub enum PzeudoNumErr {
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
    SubErr(String),
    MulErr(String),
    DivErr(String),

    // // Matmul
    Matmul2DErr(String),
    MatmulNDErr(String),

    // // View
    SlicingErr(String),
    BroadcastErr(String),
}
