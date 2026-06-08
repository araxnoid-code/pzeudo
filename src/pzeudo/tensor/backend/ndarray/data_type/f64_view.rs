use std::ops::Deref;

use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

use crate::PzeudoDataType;

pub struct F64View<A> {
    array: ArrayBase<OwnedRepr<A>, Dim<IxDynImpl>, A>,
}

// impl PzeudoDataType for F64View {}
