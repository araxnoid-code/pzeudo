#[derive(Debug)]
pub enum PzeudoErr {
    // arithmetic
    AddErr(String),
    SubErr(String),
    MulErr(String),
    DivErr(String),

    // Matmul
    Matmul2DErr(String),
    Matmul2DBackwardErr(String),

    // tools
    AbleBroadcastErr(String),

    // Tensor Method
    GetGradErr(String),
}

impl PzeudoErr {
    pub fn into_msg(self) -> String {
        match self {
            // arithmetic
            Self::AddErr(msg) => msg,
            Self::SubErr(msg) => msg,
            Self::MulErr(msg) => msg,
            Self::DivErr(msg) => msg,

            // Matmul
            PzeudoErr::Matmul2DErr(msg) => msg,
            PzeudoErr::Matmul2DBackwardErr(msg) => msg,

            // tools
            Self::AbleBroadcastErr(msg) => msg,

            // Tensor Method
            Self::GetGradErr(msg) => msg,
        }
    }
}
