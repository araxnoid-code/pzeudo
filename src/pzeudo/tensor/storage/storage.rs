use std::ops::Deref;

use ndarray::ArrayD;

use crate::{Element, GradientStorage, StorageDeref};

pub struct GradStorage<F> {
    storage: Vec<Option<ArrayD<F>>>,
    empty: Vec<usize>,
}

impl<F> GradStorage<F> {
    pub fn new(with_capacity: Option<usize>) -> GradStorage<F> {
        let storage = if let Some(with_capacity) = with_capacity {
            Vec::with_capacity(with_capacity)
        } else {
            Vec::new()
        };

        Self {
            storage,
            empty: Vec::new(),
        }
    }
}

impl<F> StorageDeref for GradStorage<F> {
    type Storage = Vec<Option<ArrayD<F>>>;
    type Empty = Vec<usize>;

    fn get_mut_storage(&mut self) -> &mut Self::Storage {
        &mut self.storage
    }

    fn get_mut_empty(&mut self) -> &mut Self::Empty {
        &mut self.empty
    }
}

impl<F> GradientStorage<ArrayD<F>> for GradStorage<F> {}
