pub trait GradientTrait {
    fn set(&mut self);
    fn mul(&mut self, rhs: &Self) -> Self;
}

///
pub trait PzeudoBackend {}
