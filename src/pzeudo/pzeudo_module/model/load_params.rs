use crate::prelude::*;

pub(crate) struct LoadParams<F> {
    pub(crate) params: Vec<Option<Vec<F>>>,
    pub(crate) idx: usize,
}

impl<F> LoadParams<F> {
    pub fn is_empty(&self) -> bool {
        self.idx >= self.params.len()
    }

    pub fn get_params(&mut self) -> Result<Vec<F>, PzeudoErr> {
        let param = Ok(self
            .params
            .get_mut(self.idx)
            .ok_or(PzeudoErr::ModuleErr(format!(
                "LoadParams::get_load_params. Index load params {} points to an invalid location.", self.idx
            )))?
            .take()
            .ok_or(PzeudoErr::ModuleErr(format!(
                "LoadParams::get_load_params. The index load params {} refers to parameters with a value of None.",
                self.idx
            )))?);

        self.idx += 1;
        param
    }
}
