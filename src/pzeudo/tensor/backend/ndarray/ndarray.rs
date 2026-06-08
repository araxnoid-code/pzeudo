use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr, RawData, ViewRepr};

use crate::{PzeudoDataType, tensor::backend::PzeudoBackend};

pub struct NDarrayBackend<P>
where
    P: PzeudoDataType,
{
    array: P,
}

impl<P> NDarrayBackend<P>
where
    P: PzeudoDataType,
{
    pub fn new(array: P) -> NDarrayBackend<P> {
        Self { array }
    }
}

impl<P> PzeudoBackend for NDarrayBackend<P>
where
    P: PzeudoDataType,
{
    fn backend(&self) -> &'static str {
        "ndarray"
    }

    fn add(&self, rhs: &Self) -> Self {
        Self {
            array: self.array.add(&rhs.array),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            array: self.array.sub(&rhs.array),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        Self {
            array: self.array.div(&rhs.array),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            array: self.array.mul(&rhs.array),
        }
    }
}
