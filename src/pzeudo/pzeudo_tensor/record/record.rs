use crate::prelude::*;

pub struct Record<F> {
    pub(crate) record: Vec<RecordLabel<F>>,
    pub(crate) skip: Vec<bool>,
}

impl<F> Record<F> {
    pub fn new() -> Record<F> {
        Self {
            record: Vec::new(),
            skip: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.record.len()
    }

    pub fn push(&mut self, label: RecordLabel<F>) {
        self.skip.push(false);
        self.record.push(label);
    }

    pub fn clear(&mut self) {
        self.record.clear();
        self.skip.clear();
    }
}
