use std::sync::{Mutex, RwLock};

use ndarray::{ArcArray, ArrayD, ArrayView, Dim, IxDynImpl};

use crate::{BackLabel, NDArrayF64, OwnTensorF64, TensorF64Inner};

pub struct ViewTensorF64<'v, Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
{
    pub(crate) array: ArrayView<'v, f64, Dim<IxDynImpl>>,
    pub(crate) grad: ArcArray<f64, Dim<IxDynImpl>>,
    back_label: Option<Mutex<BackLabel<Lhs, Rhs>>>,
}

impl<'v, Lhs, Rhs> TensorF64Inner for ViewTensorF64<'v, Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
{
    type OwnType = OwnTensorF64<Lhs, Rhs>;
    type Lhs = Lhs;
    type Rhs = Rhs;
    type ViewArr<'rhs> = ArrayView<'rhs, f64, Dim<IxDynImpl>>;
    fn view<'view>(&'view self) -> Self::ViewArr<'view> {
        self.array.view()
    }

    fn add<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType {
        let result = &self.array + rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        OwnTensorF64 {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }
    fn sub<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType {
        let result = &self.array - rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        OwnTensorF64 {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }
    fn mul<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType {
        let result = &self.array * rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        OwnTensorF64 {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }
    fn div<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self::OwnType {
        let result = &self.array / rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        OwnTensorF64 {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }

    fn get_back_label_mut(&mut self) -> &mut Option<Mutex<BackLabel<Self::Lhs, Self::Rhs>>> {
        &mut self.back_label
    }
}
