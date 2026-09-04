use std::{fs::File, io::Read};

use num_traits::Float;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::Deserialize;

use crate::{LoadParams, ModuleBuilder, PzeudoErr};

pub struct ModelBuilder<'a, F> {
    pub(crate) module: &'a mut ModuleBuilder<F>,
    pub(crate) start: usize,
    pub(crate) load_params: Option<LoadParams<F>>,
}

impl<'a, F> ModelBuilder<'a, F> {
    pub fn new(module: &'a mut ModuleBuilder<F>) -> ModelBuilder<'a, F> {
        let start = module.storage.borrow().get_params_storage().storage.len();
        Self {
            module,
            start,
            load_params: None,
        }
    }

    pub(crate) fn get_load_params(&mut self) -> Result<Option<Vec<F>>, PzeudoErr> {
        Ok(self
            .load_params
            .as_mut()
            .map_or(Ok(None), |load_params| Ok(Some(load_params.get_params()?)))?)
    }

    pub fn is_params_load(&self) -> bool {
        self.load_params.as_ref().map_or(false, |_| true)
    }

    pub(crate) fn get_load_else_generate_zeros(&mut self, len: usize) -> Result<Vec<F>, PzeudoErr>
    where
        F: Float,
        StandardNormal: Distribution<F>,
    {
        if let Some(load) = self.get_load_params()? {
            return Ok(load);
        }

        Ok(vec![F::zero(); len])
    }

    pub(crate) fn get_load_else_generate_vec(
        &mut self,
        len: usize,
        normal: &Normal<F>,
    ) -> Result<Vec<F>, PzeudoErr>
    where
        F: Float,
        StandardNormal: Distribution<F>,
    {
        if let Some(load) = self.get_load_params()? {
            return Ok(load);
        }

        Ok((0..len)
            .map(|_| normal.sample(&mut self.module.rng))
            .collect::<Vec<F>>())
    }

    /// # Load Parameters
    /// parameters will be loaded and converted using serde_json.
    /// after using ModelBuilder::load_params, the corresponding ModelBuilder will switch to load params mode instead of generate params.
    /// If the model architecture and parameters loaded in ModelBuilder are not the same, an error may occur.
    pub fn load_params(&mut self, path: &str) -> Result<(), PzeudoErr>
    where
        for<'b> F: Deserialize<'b>,
    {
        if self.load_params.is_some() {
            PzeudoErr::ModuleErr(String::from(
                "ModelBuilder::load_params. Cannot load params twice on a single ModelBuilder.",
            ));
        }

        let mut json = String::new();
        let mut file = File::open(path).map_err(|err| PzeudoErr::IOErr(err))?;
        File::read_to_string(&mut file, &mut json).map_err(|err| PzeudoErr::IOErr(err))?;

        let load_params = serde_json::from_str::<Vec<Option<Vec<F>>>>(&json)
            .map_err(|err| PzeudoErr::SerdeJsonErr(err))?;
        self.load_params = Some(LoadParams {
            params: load_params,
            idx: 0,
        });

        Ok(())
    }

    pub fn get_module(&mut self) -> &mut ModuleBuilder<F> {
        self.module
    }
}
