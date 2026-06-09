pub trait Arr {
    // fn into_arr(self) -> A;
    fn add(&self, rhs: &Self) -> Self;
}
