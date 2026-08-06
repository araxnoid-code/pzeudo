use crate::Module;

pub struct ModelBuilder<'a, F> {
    pub(crate) module: &'a mut Module<F>,
    pub(crate) start: usize,
}

impl<'a, F> ModelBuilder<'a, F> {
    pub fn new(module: &'a mut Module<F>) -> ModelBuilder<'a, F> {
        let start = module.storage.borrow().get_params_storage().storage.len();
        Self { module, start }
    }

    pub fn get_module(&mut self) -> &mut Module<F> {
        self.module
    }
}
