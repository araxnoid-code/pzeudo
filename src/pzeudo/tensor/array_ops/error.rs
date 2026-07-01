#[derive(Debug)]
pub enum PzeudoOpsErr {
    // arithmetic
    AddErr(&'static str),
    SubErr(&'static str),
    MulErr(&'static str),
    DivErr(&'static str),

    // tools
    AbleBroadcastErr(String),
}
