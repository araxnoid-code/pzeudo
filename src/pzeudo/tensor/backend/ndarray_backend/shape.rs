use crate::ShapeTrait;

pub struct NDArrayShape<'a> {
    pub shape: &'a [usize],
}

impl<'a> ShapeTrait for NDArrayShape<'a> {
    type ShapeType = &'a [usize];
    fn new(shape: Self::ShapeType) -> Self {
        Self { shape }
    }

    fn into_vec(&self) -> Vec<usize> {
        self.shape.to_vec()
    }
}
