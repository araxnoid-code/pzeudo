use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Mul},
    process::Child,
    sync::{Arc, Mutex, MutexGuard},
};

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayView, Dim, IxDynImpl, OwnedRepr, Slice, SliceArg, SliceInfo,
    SliceInfoElem, ViewRepr, linalg::Dot, s,
};
use pzeudo::Tensor;

// fn nd_matmul(lhs: &ArrayBase<>) {}

fn main() {
    // let globale:Vec<> =  vec![];
}

trait Sample<'ref_data> {
    type refr: Display;
    fn get_ref(&'ref_data self) -> Self::refr;
}

impl<'ref_data> Sample<'ref_data> for String {
    type refr = &'ref_data Self;
    fn get_ref(&'ref_data self) -> Self::refr {
        self
    }
}

struct Data<'own, 'prev_a, 'prev_b, S> {
    data: &'own S,
    prev: Option<(&'prev_a String, &'prev_b String)>,
    next: Option<String>,
}

impl<'own, 'prev_a, 'prev_b, S> Data<'own, 'prev_a, 'prev_b, S>
where
    for<'sample> S: Sample<'sample, refr = &'sample String>,
{
    fn execute1<'others, 'others_prev_a, 'others_prev_b, OS>(
        &'own mut self,
        others: &'others Data<'others, 'others_prev_a, 'others_prev_b, OS>,
    ) where
        for<'sample> OS: Sample<'sample, refr = &'sample String>,
    {
        let value = format!("{} {}", self.data.get_ref(), others.data.get_ref());
        self.next = Some(value);

        Data {
            data: self.next.as_ref().unwrap(),
            prev: Some((self.data.get_ref(), others.data.get_ref())),
            next: None,
        };
    }
}
