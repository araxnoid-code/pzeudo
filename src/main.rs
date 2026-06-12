use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Add,
    process::Child,
    sync::{Arc, Mutex, MutexGuard},
};

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayView, Dim, IxDynImpl, OwnedRepr, ViewRepr, linalg::Dot,
};
use pzeudo::{F64Base, NDArrayArr, NDArrayBackend, PzeudoBackend, PzeudoDataType, Tensor};

fn main() {
    let array_a = ArrayD::<f64>::from_elem(vec![2, 2], 1.);
    let array_b = ArrayD::<f64>::from_elem(vec![2, 2], 3.);

    let f64_base = F64Base::new(array_a);
    let ndarray_arr = NDArrayArr::new(f64_base);
    let backend = NDArrayBackend::new(ndarray_arr, None);
    let tensor_a = Tensor::new(backend, None);

    let f64_base = F64Base::new(array_b);
    let ndarray_arr = NDArrayArr::new(f64_base);
    let backend = NDArrayBackend::new(ndarray_arr, None);
    let tensor_b = Tensor::new(backend, None);

    let tensor_c = tensor_a.add(&tensor_b);
    println!("{}", tensor_c.get_array());
}

struct MyAnimal<A>
where
    A: CountryTrait,
{
    country: A,
}

trait CountryTrait {
    type Identifier<'i>: Debug
    where
        Self: 'i;

    fn identifier(&self) -> Self::Identifier<'_>;
    fn execute_something(&self) -> Self::Identifier<'_>;
}

//
struct IndonesianAnimal<I>
where
    I: IndonesianAnimalTrait,
{
    animal: I,
}
impl<I> CountryTrait for IndonesianAnimal<I>
where
    I: IndonesianAnimalTrait,
{
    type Identifier<'i>
        = I::Identifier<'i>
    where
        Self: 'i;

    fn execute_something(&self) -> Self::Identifier<'_> {
        panic!()
        // I::execute_something()
    }
    // fn execute_something(&self) -> I {
    // I::execute_something()
    // }

    fn identifier(&self) -> Self::Identifier<'_> {
        self.animal.indentifier()
    }
}

trait IndonesianAnimalTrait {
    type Identifier<'i>: Debug
    where
        Self: 'i;

    fn indentifier(&self) -> Self::Identifier<'_>;

    fn other_indentifier(&self) -> &String;

    fn create_new_animal() -> Self;

    fn execute_something() -> Self
    where
        Self: Sized,
    {
        let new_komodo = Self::create_new_animal();
        new_komodo
    }
}

struct Komodo {
    nama: String,
}

impl IndonesianAnimalTrait for Komodo {
    type Identifier<'i>
        = &'i String
    where
        Self: 'i;

    fn create_new_animal() -> Self {
        Self {
            nama: "new komodo".to_string(),
            // _phantom: PhantomData::default(),
        }
    }

    fn indentifier(&self) -> Self::Identifier<'_> {
        &self.nama
    }

    fn other_indentifier(&self) -> &String {
        &self.nama
    }
}
