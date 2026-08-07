use crate::prelude::*;

impl<F> Module<F> {
    /// ## reset
    /// will delete storage data (except params_storage) and records.
    /// functions to clean storage of data at each epoch, to prepare to store new data at the next epoch.
    pub fn reset(&self) {
        self.storage.borrow_mut().clear_storage();
        self.get_record().borrow_mut().clear();
    }

    pub fn epoch<M, T, O>(
        &self,
        mut epoch_builder: EpochBuilder<M, T>,
        f: fn(usize, &Module<F>, &mut M, &T) -> Result<O, PzeudoErr>,
    ) -> Result<(), PzeudoErr> {
        for i in 0..epoch_builder.epoch {
            f(i, self, &mut epoch_builder.model, &epoch_builder.arg)?;

            // RESET
            self.reset();
        }

        Ok(())
    }
}
