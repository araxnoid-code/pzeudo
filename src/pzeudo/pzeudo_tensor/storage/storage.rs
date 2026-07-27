use std::{format, panic};

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

    // pub fn get_element(&self, idx: StorageType) -> Result<GetElementOutput<'_, F>, PzeudoErr> {
    //     let data = match idx {
    //         StorageType::Permanent(idx) => {
    //             let data = self
    //                 .permanent_storage
    //                 .get(idx)
    //                 .ok_or(PzeudoErr::StorageGetErr(format!(
    //                     "ArrayStorage::get. index {idx} points to an invalid location on storage."
    //                 )))?;
    //             GetElementOutput::Permanent(data)
    //         }
    //         StorageType::Storage(idx) => {
    //             let data = self
    //                 .arr_storage
    //                 .get(idx)
    //                 .ok_or(PzeudoErr::StorageGetErr(format!(
    //                     "ArrayStorage::get. index {idx} points to an invalid location on storage."
    //                 )))?
    //                 .as_ref()
    //                 .ok_or(PzeudoErr::StorageGetErr(format!(
    //                     "ArrayStorage::get. index {idx} points to elements that have the value None in storage."
    //                 )))?;
    //             GetElementOutput::Storage(data)
    //         }
    //     };

    //     Ok(data)
    // }

    // pub fn get_arr_element_mut(
    //     &mut self,
    //     idx: StorageType,
    // ) -> Result<GetElementMutOutput<'_, F>, PzeudoErr> {
    //     let data = match idx {
    //         StorageType::Permanent(idx) => {
    //             let data = self
    //                 .permanent_storage
    //                 .get_mut(idx)
    //                 .ok_or(PzeudoErr::StorageGetErr(format!(
    //                     "ArrayStorage::get. index {idx} points to an invalid location on storage."
    //                 )))?;
    //             GetElementMutOutput::Permanent(data)
    //         }
    //         StorageType::Storage(idx) => {
    //             let data = self
    //                 .arr_storage
    //                 .get_mut(idx)
    //                 .ok_or(PzeudoErr::StorageGetErr(format!(
    //                     "ArrayStorage::get. index {idx} points to an invalid location on storage."
    //                 )))?
    //                 .as_mut()
    //                 .ok_or(PzeudoErr::StorageGetErr(format!(
    //                     "ArrayStorage::get. index {idx} points to elements that have the value None in storage."
    //                 )))?;
    //             GetElementMutOutput::Storage(data)
    //         }
    //     };

    //     Ok(data)
    // }

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

    // pub fn get_as_array_ref<T>(
    //     &self,
    //     idx: StorageType,
    //     arr_contiguous_type: ContiguousType,
    // ) -> Result<ArrayRef<'_, F, T>, PzeudoErr> {
    //     match idx {
    //         StorageType::Permanent(idx) => {
    //             let permanent = self.permanent_storage
    //                 .get(idx)
    //                 .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
    //                     "ArrayStorage::get_as_array_ref. Index of type permanent_storage points to {idx} which is an invalid location in storage"
    //                 )))?;

    //             let array = match arr_contiguous_type {
    //                 ContiguousType::Arr => &permanent.array,
    //                 ContiguousType::Grad => &permanent.grad,
    //             };

    //             return Ok(ArrayRef {
    //                 data: &array.data,
    //                 offset: array.offset,
    //                 shape: &array.shape,
    //                 stride: &array.stride,
    //                 _array_type: Default::default(),
    //             });
    //         }
    //         StorageType::Storage(idx) => match arr_contiguous_type {
    //             ContiguousType::Arr => {
    //                 let element = self.arr_storage.get_arr(idx)?;
    //                 match element {
    //                     ElementType::Contiguous(array, _) => {
    //                         return Ok(ArrayRef {
    //                             data: &array.data,
    //                             offset: array.offset,
    //                             shape: &array.shape,
    //                             stride: &array.stride,
    //                             _array_type: Default::default(),
    //                         });
    //                     }
    //                     ElementType::View(p_idx, metadata) => match p_idx {
    //                         StorageType::Permanent(p_idx) => {
    //                             let permanent = self.permanent_storage
    //                                         .get(*p_idx)
    //                                         .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
    //                                             "ArrayStorage::get_as_array_ref. index {idx} points to a view that has index {p_idx} that points to permanent_storage, but index {p_idx} is not a valid index."
    //                                         )))?;

    //                             let array = match arr_contiguous_type {
    //                                 ContiguousType::Arr => &permanent.array,
    //                                 ContiguousType::Grad => &permanent.grad,
    //                             };

    //                             return Ok(ArrayRef {
    //                                 data: &array.data,
    //                                 offset: metadata.offset,
    //                                 shape: &metadata.shape,
    //                                 stride: &metadata.stride,
    //                                 _array_type: Default::default(),
    //                             });
    //                         }
    //                         StorageType::Storage(p_idx) => {
    //                             let element = self.arr_storage.get_arr(*p_idx)?;
    //                             match element {
    //                                 ElementType::View(_, _) => {
    //                                     return Err(PzeudoErr::StorageGetAsArrayRefErr(format!(
    //                                         "ArrayStorage::get_as_array_ref. index {idx} points to the View element that has index {p_idx} which points to the element that has value View Also, View pointing to View is prohibited"
    //                                     )));
    //                                 }
    //                                 ElementType::Contiguous(array, _) => {
    //                                     return Ok(ArrayRef {
    //                                         data: &array.data,
    //                                         offset: metadata.offset,
    //                                         shape: &metadata.shape,
    //                                         stride: &metadata.stride,
    //                                         _array_type: Default::default(),
    //                                     });
    //                                 }
    //                             }
    //                         }
    //                     },
    //                 }
    //             }
    //             ContiguousType::Grad => {
    //                 let grad = self.grad_storage.get_grad(idx)?;
    //                 return Ok(ArrayRef {
    //                     data: &grad.data,
    //                     offset: grad.offset,
    //                     shape: &grad.shape,
    //                     stride: &grad.stride,
    //                     _array_type: Default::default(),
    //                 });
    //             }
    //         },
    //     }
    // }

    // pub fn get_as_array_ref_mut(
    //     &mut self,
    //     idx: StorageType,
    //     arr_contiguous_type: ContiguousType,
    // ) -> Result<ArrayRefMut<'_, F, Contiguous>, PzeudoErr> {
    //     match idx {
    //         StorageType::Permanent(idx) => {
    //             let permanent = self.permanent_storage
    //                 .get_mut(idx)
    //                 .ok_or(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
    //                     "ArrayStorage::get_as_array_ref. Index of type permanent_storage points to {idx} which is an invalid location in storage"
    //                 )))?;

    //             let array = match arr_contiguous_type {
    //                 ContiguousType::Arr => &mut permanent.array,
    //                 ContiguousType::Grad => &mut permanent.grad,
    //             };

    //             return Ok(ArrayRefMut {
    //                 data: &mut array.data,
    //                 offset: array.offset,
    //                 shape: &array.shape,
    //                 stride: &array.stride,
    //                 _array_type: Default::default(),
    //             });
    //         }
    //         StorageType::Storage(idx) => match arr_contiguous_type {
    //             ContiguousType::Arr => {
    //                 let element = self.arr_storage.get_arr_mut(idx)?;
    //                 match element {
    //                     ElementType::Contiguous(array, _) => {
    //                         return Ok(ArrayRefMut {
    //                             data: &mut array.data,
    //                             offset: array.offset,
    //                             shape: &array.shape,
    //                             stride: &array.stride,
    //                             _array_type: Default::default(),
    //                         });
    //                     }
    //                     ElementType::View(_, _) => {
    //                         return Err(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
    //                             "ArrayStorage::get_as_array_ref_mut. The index {idx} points to the View element, the View element cannot be changed (mut)"
    //                         )));
    //                     }
    //                 }
    //             }
    //             ContiguousType::Grad => {
    //                 let grad = self.grad_storage.get_grad_mut(idx)?;
    //                 return Ok(ArrayRefMut {
    //                     data: &mut grad.data,
    //                     offset: grad.offset,
    //                     shape: &grad.shape,
    //                     stride: &grad.stride,
    //                     _array_type: Default::default(),
    //                 });
    //             }
    //         },
    //     }
    // }

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
