use crate::prelude::*;

impl<F> Module<F> {
    pub fn epoch<M, T>(
        &self,
        epoch_builder: EpochBuilder<M, T>,
        f: fn(usize, &Module<F>, &M, &T),
    ) -> Result<(), PzeudoErr> {
        for i in 0..epoch_builder.epoch {
            f(i, self, &epoch_builder.model, &epoch_builder.arg);

            // RESET
            self.storage.borrow_mut().clear_storage();
        }

        Ok(())
    }
}
