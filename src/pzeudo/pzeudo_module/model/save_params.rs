use serde::Serialize;

use crate::prelude::*;
use std::{cell::RefCell, fs::File, io::Write, rc::Rc};

pub trait SaveParamsTrait<F>
where
    F: Serialize,
{
    fn get_storage(&self) -> &Rc<RefCell<ArrayStorage<F>>>;
    fn get_range(&self) -> (usize, usize);

    /// ## Save Parameters
    /// Parameters will be saved using serde_json.
    /// parameters stored in the form of a flat array.
    fn save_params(&self, path: &str) -> Result<(), PzeudoErr> {
        let storage = self.get_storage().borrow();
        let range = self.get_range();
        let params_storage = &storage.get_params_storage().storage[range.0..range.1];

        let mut saving_params = Vec::with_capacity(params_storage.len());
        for params in params_storage {
            saving_params.push(Some(&params.array.data));
        }

        let json =
            serde_json::to_string(&saving_params).map_err(|err| PzeudoErr::SerdeJsonErr(err))?;

        let mut file = File::create(path).map_err(|err| PzeudoErr::IOErr(err))?;
        file.write_all(json.as_bytes())
            .map_err(|err| PzeudoErr::IOErr(err))?;

        Ok(())
    }
}
