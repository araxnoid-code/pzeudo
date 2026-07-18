use std::{cell::RefCell, rc::Rc};

use crate::prelude::*;

pub enum TensorType<'a, T> {
    Contiguous,
    View(&'a TensorMetadata),
    _ArrayType(T),
}

pub trait TensorTrait<F, T> {
    fn get_array_idx(&self) -> usize;
    fn get_grad_idx(&self) -> Option<usize>;
    fn get_array_data(&self) -> (usize, TensorType<'_, T>);

    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>>;
    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>>;
}

impl<F> TensorTrait<F, Contiguous> for Tensor<F> {
    fn get_array_data(&self) -> (usize, TensorType<'_, Contiguous>) {
        (self.array_idx, TensorType::Contiguous)
    }

    fn get_array_idx(&self) -> usize {
        self.array_idx
    }

    fn get_grad_idx(&self) -> Option<usize> {
        self.grad_idx
    }

    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>> {
        &self.record
    }

    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }
}

impl<F> TensorTrait<F, View> for TensorView<F> {
    fn get_array_data(&self) -> (usize, TensorType<'_, View>) {
        (self.array_idx, TensorType::View(&self.metadata))
    }

    fn get_array_idx(&self) -> usize {
        self.array_idx
    }

    fn get_grad_idx(&self) -> Option<usize> {
        self.grad_idx
    }

    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>> {
        &self.record
    }

    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>> {
        &self.storage
    }
}
