use std::{cell::RefCell, rc::Rc};

use crate::prelude::*;

pub trait TensorTrait<F> {
    fn get_array_idx(&self) -> usize;
    fn get_array_metadata(&self) -> &TensorMetadata;
    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>>;
    fn get_record(&self) -> &Rc<RefCell<Vec<RecordLabel>>>;
}
