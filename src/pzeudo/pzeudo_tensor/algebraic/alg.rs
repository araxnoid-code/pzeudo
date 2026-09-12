pub trait AlgebraicAble {
    fn _algebraic_add(self, rhs: Self) -> Self
    where
        Self: Sized;

    fn _algebraic_sub(self, rhs: Self) -> Self
    where
        Self: Sized;

    fn _algebraic_mul(self, rhs: Self) -> Self
    where
        Self: Sized;

    fn _algebraic_div(self, rhs: Self) -> Self
    where
        Self: Sized;

    fn _algebraic_rem(self, rhs: Self) -> Self
    where
        Self: Sized;
}

impl AlgebraicAble for f32 {
    #[inline]
    fn _algebraic_add(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_add(rhs)
    }

    #[inline]
    fn _algebraic_sub(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_sub(rhs)
    }

    #[inline]
    fn _algebraic_mul(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_mul(rhs)
    }

    #[inline]
    fn _algebraic_div(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_div(rhs)
    }

    #[inline]
    fn _algebraic_rem(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_rem(rhs)
    }
}

impl AlgebraicAble for f64 {
    #[inline]
    fn _algebraic_add(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_add(rhs)
    }

    #[inline]
    fn _algebraic_sub(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_sub(rhs)
    }

    #[inline]
    fn _algebraic_mul(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_mul(rhs)
    }

    #[inline]
    fn _algebraic_div(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_div(rhs)
    }

    #[inline]
    fn _algebraic_rem(self, rhs: Self) -> Self
    where
        Self: Sized,
    {
        self.algebraic_rem(rhs)
    }
}
