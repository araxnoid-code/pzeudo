use crate::{prelude::*, pzeudo_module::module::Module};

impl<F, M> Module<F, M> {
    pub fn reset(&self) {
        self.clear_storage();
        self.get_record().borrow_mut().clear();
    }

    pub fn epoch<T, O>(
        &mut self,
        epoch_builder: EpochBuilder<T>,
        f: fn(usize, &Module<F, M>, &mut M, &T) -> Result<O, PzeudoErr>,
    ) -> Result<(), PzeudoErr> {
        let mut model = self.model.take().unwrap();
        for i in 0..epoch_builder.epoch {
            f(i, self, &mut model, &epoch_builder.arg)?;

            // RESET
            self.reset();
        }
        self.model = Some(model);
        Ok(())
    }
}
