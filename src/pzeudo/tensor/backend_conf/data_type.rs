pub enum PzeudoDataType {
    F64(f64),
    I32(i32),
}

pub trait PzeudoDataTypeTrait {
    fn into_pzeudo_data_type(&self) -> PzeudoDataType;
}

impl PzeudoDataTypeTrait for f64 {
    fn into_pzeudo_data_type(&self) -> PzeudoDataType {
        PzeudoDataType::F64(*self)
    }
}

impl PzeudoDataTypeTrait for i32 {
    fn into_pzeudo_data_type(&self) -> PzeudoDataType {
        PzeudoDataType::I32(*self)
    }
}
