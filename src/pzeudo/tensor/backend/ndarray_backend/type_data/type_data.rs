pub trait NDArrayDataType {
    fn add(&self, rhs: &Self) -> Self;
}
