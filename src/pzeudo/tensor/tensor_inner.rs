use std::{io::Seek, sync::Mutex};

use crate::BackLabel;

pub trait TensorF64Inner {
    type OwnType: TensorF64Inner;
    type ViewArr<'view>;
    type Lhs: TensorF64Inner;
    type Rhs: TensorF64Inner;

    fn view<'view>(&'view self) -> Self::ViewArr<'view>;
    fn own_type_into(own_type: Self::OwnType) -> Self;

    fn add<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType;
    fn sub<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType;
    fn mul<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType;
    fn div<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType;

    fn get_back_label_mut(&mut self) -> &mut Option<Mutex<BackLabel<Self::Lhs, Self::Rhs>>>;
}
