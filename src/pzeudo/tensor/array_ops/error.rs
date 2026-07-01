#[derive(Debug)]
pub enum PzeudoErr {
    // arithmetic
    AddErr(String),
    SubErr(String),
    MulErr(String),
    DivErr(String),

    // tools
    AbleBroadcastErr(String),
}

impl PzeudoErr {
    pub fn into_msg(self) -> String {
        match self {
            // arithmetic
            Self::AddErr(msg) => msg,
            Self::SubErr(msg) => msg,
            Self::MulErr(msg) => msg,
            Self::DivErr(msg) => msg,

            // tools
            Self::AbleBroadcastErr(msg) => msg,
        }
    }
}
