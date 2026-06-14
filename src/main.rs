use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
    process::Child,
    sync::{Arc, Mutex, MutexGuard},
};

use ndarray::{ArcArrayD, linalg::Dot};
// use pzeudo::{OwnTensorF64, TensorF64Empty};

// fn nd_matmul(lhs: &ArrayBase<>) {}

struct DataArr<A, Rhs>
where
    A: Add + Mul + Div + Sub + Dot<Rhs> + Sync,
    Rhs: Dot<Self>,
{
    array: A,
    _pahtom: PhantomData<Rhs>,
}

fn main() {
    let array = ArcArrayD::<f64>::zeros(vec![10]);

    let data = DataArr {
        array: array,
        _pahtom: Default::default(),
    };
    // let tensor: OwnTensorF64<TensorF64Empty, TensorF64Empty> = OwnTensorF64::new(array, None, None);
}

struct Zoo<A> {
    animal: A,
}

trait AnimalTrait {
    type Lhs;
    type Rhs;

    fn get_teman(&self) -> &Option<(Self::Lhs, Self::Rhs)>;
}

struct Dummy;

impl AnimalTrait for Dummy {
    type Lhs = Dummy;
    type Rhs = Dummy;

    fn get_teman(&self) -> &Option<(Self::Lhs, Self::Rhs)> {
        &None
    }
}

struct Kucing<Lhs, Rhs>
where
    Lhs: AnimalTrait,
    Rhs: AnimalTrait,
{
    data: String,
    teman: Option<(Lhs, Rhs)>,
}

impl<Lhs, Rhs> AnimalTrait for Kucing<Lhs, Rhs>
where
    Lhs: AnimalTrait,
    Rhs: AnimalTrait,
{
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn get_teman(&self) -> &Option<(Self::Lhs, Self::Rhs)> {
        &self.teman
    }
}
