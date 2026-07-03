use crate::ArrayTrait;

trait ArrayAdd {
    type Output: ArrayTrait;
    fn _add(&self) -> Self::Output;
}
