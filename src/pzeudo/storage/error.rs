#[derive(Debug)]
pub enum PzeudoStorageErr {
    IndexNotFoundErr(String),
    PushElementErr(String),
    RemoveELementErr(String),
    GetElementErr(String),
}

impl PzeudoStorageErr {
    pub fn to_string(self) -> String {
        match self {
            Self::PushElementErr(err) => err,
            Self::IndexNotFoundErr(err) => err,
            Self::RemoveELementErr(err) => err,
            Self::GetElementErr(err) => err,
        }
    }
}
