use std::ops::Deref;

use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

use crate::PzeudoDataType;

pub struct F64View {
    array: ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>, i64>,
}

// impl PzeudoDataType for F64View {}
