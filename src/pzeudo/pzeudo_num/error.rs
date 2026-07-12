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
}
