pub trait ShapeTrait {
    type ShapeType;
    fn new(shape: Self::ShapeType) -> Self;
}

pub trait Arr<'s> {
    type ShapeType: ShapeTrait;

    // desc
    fn get_shape(&'s self) -> Self::ShapeType;

    // element wise-ops
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;
}
