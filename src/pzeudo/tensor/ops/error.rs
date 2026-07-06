#[derive(Debug)]
pub enum PzeudoOpsErr {
    // Arith
    AddErr(String),
    AddBackwardErr(String),

    // Transform
    BroadcastErr(String),
}
