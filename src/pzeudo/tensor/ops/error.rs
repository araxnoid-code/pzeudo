#[derive(Debug)]
pub enum PzeudoOpsErr {
    // Arith
    AddErr(String),
    AddBackwardErr(String),

    DivErr(String),
    DivBackwardErr(String),

    MulErr(String),
    MulBackwardErr(String),

    Sub(String),
    SubBackwardErr(String),

    // Matmul
    Matmul2dErr(String),
    Matmul2dBackwardErr(String),

    // Transform
    BroadcastErr(String),

    ViewErr(String),
    ViewBackwardErr(String),
}
