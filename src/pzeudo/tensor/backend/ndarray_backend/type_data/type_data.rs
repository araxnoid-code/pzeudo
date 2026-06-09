pub trait NDArrayDataType {
    // desc
    fn get_shape(&self) -> &[usize];

    // initial
    fn zeros(shape: &[usize]) -> Self;
    fn ones(shape: &[usize]) -> Self;

    // element-wise ops
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;
}
