use std::format;

use crate::{
    PzeudoErr::StorageNoGradErr,
    prelude::*,
    pzeudo_tensor::storage::component::{ArrStorage, GradStorage, ViewStorage},
};

pub struct ArrayStorage<F> {
    pub(crate) params_storage: ParamsStorage<F>,
    pub(crate) grad_storage: GradStorage<F>,
    pub(crate) arr_storage: ArrStorage<F>,
    pub(crate) view_storage: ViewStorage,
}

impl<F> ArrayStorage<F> {
    pub fn new(capacity: Option<usize>) -> ArrayStorage<F> {
        Self {
            params_storage: ParamsStorage::new(capacity.unwrap_or(0)),
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

    pub fn get_params_storage(&self) -> &ParamsStorage<F> {
        &self.params_storage
    }

    pub fn get_params_storage_mut(&mut self) -> &mut ParamsStorage<F> {
        &mut self.params_storage
    }

    pub fn get_grad_storage(&self) -> &GradStorage<F> {
        &self.grad_storage
    }

    pub fn get_grad_storage_mut(&mut self) -> &mut ArrStorage<F> {
        &mut self.arr_storage
    }

    pub fn get_grad_element_mut(&mut self, idx: StorageType) -> Result<&mut Array<F>, PzeudoErr> {
        let data = match idx {
            StorageType::Param(idx) => {
                let data = self
                    .params_storage
                    .storage
                    .get_mut(idx)
                    .ok_or(PzeudoErr::StorageErr(format!(
                        "ArrayStorage::get_grad_element_mut. index {idx} points to an invalid location on storage."
                    )))?;
                data.grad.as_mut().ok_or(StorageNoGradErr(format!(
                    "ArrayStorage::get_grad_element_mut. permanent array with index {idx} has status NoGrad"
                )))?
            }
            StorageType::Arr(idx, grad_time) => {
                let data = self.grad_storage.get_grad_mut(
                    idx,
                    grad_time.ok_or(PzeudoErr::StorageErr(format!(
                        "ArrayStorage::get_grad_element_mut. Cannot access gradient using StorageGrad::get_grad_mut method if grad_time is None"
                    )))?
                )?;
                data
            }
            StorageType::View(_) => {
                return Err(PzeudoErr::StorageErr(format!(
                    "ArrayStorage::get_grad_element_mut. saat ini Tidak bisa mengambil gradient dari view"
                )));
            }
        };

        Ok(data)
    }

    pub fn clear_storage(&mut self) {
        self.arr_storage.clear();
        self.grad_storage.clear();
        self.view_storage.clear();
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
