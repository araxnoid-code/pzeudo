pub trait EmbbedingIndex {
    fn into_usize(&self) -> usize;
}

impl EmbbedingIndex for f32 {
    fn into_usize(&self) -> usize {
        *self as usize
    }
}

impl EmbbedingIndex for f64 {
    fn into_usize(&self) -> usize {
        *self as usize
    }
}
