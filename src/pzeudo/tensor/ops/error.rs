#[derive(Debug)]
pub enum PzeudoOpsErr {
    // Arith
    AddErr(String),
    AddBackwardErr(String),

    DivErr(String),
    DivBackwardErr(String),

    MulErr(String),
    MulBackwardErr(String),

    // Transform
    BroadcastErr(String),
}
