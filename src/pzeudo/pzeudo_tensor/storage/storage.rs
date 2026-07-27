use std::format;

use crate::{
    prelude::*,
    pzeudo_tensor::storage::compunent::{ArrStorage, GradStorage, ViewStorage},
};

pub struct ArrayStorage<F> {
    pub(crate) permanent_storage: Vec<PermanentTensor<F>>,
    pub(crate) grad_storage: GradStorage<F>,
    pub(crate) arr_storage: ArrStorage<F>,
    pub(crate) view_storage: ViewStorage,
}

impl<F> ArrayStorage<F> {
    pub fn new(capacity: Option<usize>) -> ArrayStorage<F> {
        Self {
            permanent_storage: Vec::new(),
            grad_storage: GradStorage::new(capacity),
            arr_storage: ArrStorage::new(capacity),
            view_storage: ViewStorage::new(capacity),
        }
    }

    pub fn get_arr_storage(&self) -> &ArrStorage<F> {
        &self.arr_storage
    }

    pub fn get_arr_storage_mut(&mut self) -> &mut ArrStorage<F> {
        &mut self.arr_storage
    }

    pub fn get_permanent_storage(&self) -> &Vec<PermanentTensor<F>> {
        &self.permanent_storage
    }

    pub fn get_permanent_storage_mut(&mut self) -> &mut Vec<PermanentTensor<F>> {
        &mut self.permanent_storage
    }

    pub fn get_grad_storage(&self) -> &GradStorage<F> {
        &self.grad_storage
    }

    pub fn get_grad_storage_mut(&mut self) -> &mut ArrStorage<F> {
        &mut self.arr_storage
    }

    pub fn get_grad_element_mut(&mut self, idx: StorageType) -> Result<&mut Array<F>, PzeudoErr> {
        let data = match idx {
            StorageType::Permanent(idx) => {
                let data = self
                    .permanent_storage
                    .get_mut(idx)
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get_grad_element_mut. index {idx} points to an invalid location on storage."
                    )))?;
                &mut data.grad
            }
            StorageType::Arr(idx) => {
                let data = self.grad_storage.get_grad_mut(idx)?;
                data
            }
            StorageType::View(_) => {
                return Err(PzeudoErr::StorageGetErr(format!(
                    "ArrayStorage::get_grad_element_mut. saat ini Tidak bisa mengambil gradient dari view"
                )));
            }
        };

        Ok(data)
    }

    pub fn clear_storage(&mut self) {
        self.arr_storage.clear();
        self.grad_storage.clear();
    }
}

pub trait StorageF32F64 {
    fn to_mut_f32(&mut self) -> Option<&mut ArrayStorage<f32>>;
    fn to_mut_f64(&mut self) -> Option<&mut ArrayStorage<f64>>;
}

impl StorageF32F64 for ArrayStorage<f32> {
    fn to_mut_f32(&mut self) -> Option<&mut ArrayStorage<f32>> {
        Some(self)
    }

    fn to_mut_f64(&mut self) -> Option<&mut ArrayStorage<f64>> {
        None
    }
}

impl StorageF32F64 for ArrayStorage<f64> {
    fn to_mut_f32(&mut self) -> Option<&mut ArrayStorage<f32>> {
        None
    }

    fn to_mut_f64(&mut self) -> Option<&mut ArrayStorage<f64>> {
        Some(self)
    }
}
