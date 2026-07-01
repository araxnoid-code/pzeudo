use ndarray::ArrayD;

pub trait OwnAble {}
impl OwnAble for ArrayD<f32> {}
