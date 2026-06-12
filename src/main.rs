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
    let data = MyAnimal {
        country: IndonesianAnimal {
            animal: Komodo {
                nama: "komodo dragon".to_string(),
            },
        },
    };

    data.country.execute_something();
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
