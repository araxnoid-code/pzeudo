use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr, RawData, ViewRepr};

use crate::{PzeudoDataType, tensor::backend::PzeudoBackend};

pub struct NDarrayBackend<P>
where
    P: PzeudoDataType,
{
    data: P,
}

impl<P> NDarrayBackend<P>
where
    P: PzeudoDataType,
{
    pub fn new(data: P) -> NDarrayBackend<P> {
        Self { data }
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
            data: self.data.add(&rhs.data),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            data: self.data.sub(&rhs.data),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        Self {
            data: self.data.div(&rhs.data),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            data: self.data.mul(&rhs.data),
        }
    }
}
