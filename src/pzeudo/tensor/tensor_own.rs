use std::{
    ops::Add,
    sync::{Arc, Mutex, RwLock},
};

use ndarray::{ArcArray, Array, ArrayD, ArrayView, Dim, IxDynImpl, s};

use crate::{BackLabel, NDArrayF64, TensorF64Inner};

pub struct OwnTensorF64<Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
{
    pub(crate) array: ArcArray<f64, Dim<IxDynImpl>>,
    pub(crate) grad: Option<RwLock<Array<f64, Dim<IxDynImpl>>>>,
    pub(crate) back_label: Option<Mutex<BackLabel<Lhs, Rhs>>>,
}

impl<Lhs, Rhs> OwnTensorF64<Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
{
    pub fn new(
        array: ArcArray<f64, Dim<IxDynImpl>>,
        grad: Option<Array<f64, Dim<IxDynImpl>>>,
        back_label: Option<BackLabel<Lhs, Rhs>>,
    ) -> OwnTensorF64<Lhs, Rhs> {
        Self {
            array,
            grad: grad.map(|grad| RwLock::new(grad)),
            back_label: back_label.map(|label| Mutex::new(label)),
        }
    }
}

impl<Lhs, Rhs> TensorF64Inner for OwnTensorF64<Lhs, Rhs>
where
    Lhs: TensorF64Inner,
    Rhs: TensorF64Inner,
{
    type OwnType = Self;
    type ViewArr<'rhs> = ArrayView<'rhs, f64, Dim<IxDynImpl>>;
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn view<'view>(&'view self) -> Self::ViewArr<'view> {
        self.array.view()
    }

    fn own_type_into(own_type: Self::OwnType) -> Self {
        own_type
    }

    fn add<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        let result = &self.array + rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        Self {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }
    fn sub<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        let result = &self.array - rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        Self {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }
    fn mul<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        let result = &self.array * rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        Self {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }
    fn div<'rhs>(&self, rhs: &Self::ViewArr<'rhs>) -> Self {
        let result = &self.array / rhs;
        let grad = Some(RwLock::new(ArrayD::<f64>::zeros(result.shape())));
        Self {
            array: result.to_shared(),
            grad,
            back_label: None,
        }
    }

    fn get_back_label_mut(&mut self) -> &mut Option<Mutex<BackLabel<Self::Lhs, Self::Rhs>>> {
        &mut self.back_label
    }
}
