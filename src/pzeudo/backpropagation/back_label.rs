use std::sync::{Arc, RwLock};

use crate::{NDArrayF64, OwnTensorF64, TensorF64Inner};

pub enum BackLabel<Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
{
    Add(Arc<Lhs>, Arc<Rhs>),
    Sub(Arc<Lhs>, Arc<Rhs>),
    Mul(Arc<Lhs>, Arc<Rhs>),
    Div(Arc<Lhs>, Arc<Rhs>),
}

pub struct TensorF64Empty;
impl TensorF64Inner for TensorF64Empty {
    type OwnType = TensorF64Empty;
    type Lhs = TensorF64Empty;
    type Rhs = TensorF64Empty;
    type ViewArr<'view> = TensorF64Empty;
    fn add<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        panic!()
    }
    fn div<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        panic!()
    }
    fn get_back_label_mut(
        &mut self,
    ) -> &mut Option<std::sync::Mutex<BackLabel<Self::Lhs, Self::Rhs>>> {
        panic!()
    }
    fn mul<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        panic!()
    }
    fn sub<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        panic!()
    }
    fn view<'view>(&'view self) -> Self::ViewArr<'view> {
        panic!()
    }
}
