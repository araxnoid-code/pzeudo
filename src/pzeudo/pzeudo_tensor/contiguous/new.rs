use crate::prelude::*;
use std::{cell::RefCell, rc::Rc};

impl<F> Tensor<F> {
    fn new(
        array: Array<F>,
        gradient: Option<Array<F>>,
        storage: Rc<RefCell<ArrayStorage<F>>>,
    ) -> Result<Self, PzeudoErr> {
        let mut storage_mut = storage.borrow_mut();
        let array_idx = storage_mut.push(array)?;
        let grad_idx =
            gradient.map_or(Ok(None), |grad| storage_mut.push(grad).map(|idx| Some(idx)))?;

        drop(storage_mut);

        let tensor = Self {
            array_idx,
            grad_idx,
            storage,
        };

        Ok(tensor)
    }
}
