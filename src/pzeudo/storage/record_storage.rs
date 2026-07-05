use crate::StorageTrait;
use ndarray::ArrayD;

pub struct RecordStorage<F> {
    storage: Vec<Option<ArrayD<F>>>,
    empty_idx: Vec<usize>,
}

impl<F> RecordStorage<F> {
    fn new(with_capacity: Option<usize>) -> RecordStorage<F> {
        Self {
            storage: with_capacity.map_or(Vec::new(), |capacity| Vec::with_capacity(capacity)),
            empty_idx: Vec::new(),
        }
    }
}

impl<F> StorageTrait<ArrayD<F>> for RecordStorage<F> {
    fn get_storage(&self) -> &Vec<Option<ArrayD<F>>> {
        &self.storage
    }

    fn get_empty_idx(&self) -> &Vec<usize> {
        &self.empty_idx
    }

    fn get_mut_storage(&mut self) -> &mut Vec<Option<ArrayD<F>>> {
        &mut self.storage
    }

    fn get_mut_empty_idx(&mut self) -> &mut Vec<usize> {
        &mut self.empty_idx
    }
}
