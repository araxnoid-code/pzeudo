pub trait GradientTrait {
    type View<'a>
    where
        Self: 'a;

    fn to_ones(&mut self);
    fn add_assign(&mut self, assign: Self::View<'_>);
    fn to_view(&self) -> Self::View<'_>;
}
