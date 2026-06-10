pub trait ShapeTrait {
    type ShapeType;
    fn new(shape: Self::ShapeType) -> Self;
    // fn get_shape(&self) -> Self::ShapeType;
}
