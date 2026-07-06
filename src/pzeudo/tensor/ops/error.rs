#[derive(Debug)]
pub enum PzeudoOpsErr {
    // Arith
    AddErr(String),
    AddBackwardErr(String),

    DivErr(String),
    DivBackwardErr(String),

    // Transform
    BroadcastErr(String),
}
