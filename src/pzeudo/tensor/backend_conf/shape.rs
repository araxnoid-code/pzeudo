pub trait ShapeTrait {
    type ShapeType;
    fn new(shape: Self::ShapeType) -> Self;
    fn into_vec(&self) -> Vec<usize>;
    // fn get_shape(&self) -> Self::ShapeType;
}
