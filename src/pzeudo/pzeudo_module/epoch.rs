use std::format;

use crate::prelude::*;

impl<F> Module<F> {
    pub fn epoch<M, O>(
        &self,
        epoch: usize,
        model: M,
        other: O,
        f: fn(usize, &Module<F>, &M, &O),
    ) -> Result<(), PzeudoErr> {
        for i in 0..epoch {
            f(i, self, &model, &other);

            // RESET
            let mut storage = self.storage.borrow_mut().clear_storage();
        }

        Ok(())
    }
}
