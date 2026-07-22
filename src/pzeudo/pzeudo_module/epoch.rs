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
        let mut permanent_list = Vec::new();
        for element in self.storage.borrow().get_storage() {
            if let Some(element) = element {
                if let ElementType::PermanentTensor(idx) = element {
                    permanent_list.push(Some(ElementType::<F>::PermanentTensor(*idx)));
                } else {
                    return Err(PzeudoErr::EpochErr(format!(
                        "Module::epoch. In storage, there are tensors that are not permanent. Before entering an epoch, all tensors must be permanent.",
                    )));
                }
            } else {
                return Err(PzeudoErr::EpochErr(format!(
                    "Module::epoch. in storage, there is a None value, make sure no tensors are deleted before the Module runs the epoch.",
                )));
            }
        }

        for i in 0..epoch {
            f(i, self, &model, &other);

            // RESET
            let mut storage = self.storage.borrow_mut();
            storage.clear_storage();
            storage.get_mut_storage().extend_from_slice(&permanent_list);
        }

        Ok(())
    }
}
