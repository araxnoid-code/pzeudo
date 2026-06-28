use std::{
    ops::Add,
    sync::{Arc, RwLock},
};

use ndarray::{Array, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr, RawData};

fn main() {
    // pub struct ArrayBase<S, D, A = <S as RawData>::Elem>
    // where S: RawData<Elem = A>
    let array: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> = ArrayD::<f32>::zeros(vec![1]);
    // let data = OwnedRepr<f32>;
}

struct Data<A, S = <A as RawData>::Elem>
where
    A: RawData<Elem = S>,
{
    arr: ArrayD<A>,
}

// impl<A, S> Data<A, S>
// where
//     A: RawData<Elem = S>,
// {
//     fn something(&self) {
//         self.arr.view().add(1.);
//     }
// }
