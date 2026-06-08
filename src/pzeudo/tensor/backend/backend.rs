///
pub trait PzeudoBackend {
    /// backend identifier
    fn backend(&self) -> &'static str;

    // base operation
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
}
