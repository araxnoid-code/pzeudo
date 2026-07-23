use std::format;

use crate::{prelude::*, pzeudo_tensor::storage::helper::storage_contiguous_type_check};

pub struct ArrayStorage<F> {
    permanent_storage: Vec<PermanentTensor<F>>,
    storage: Vec<Option<ElementType<F>>>,
    empty_idx: Vec<usize>,
}

impl<F> ArrayStorage<F> {
    pub fn new(capacity: Option<usize>) -> ArrayStorage<F> {
        let len = capacity.unwrap_or(1);
        Self {
            permanent_storage: Vec::new(),
            storage: Vec::with_capacity(len),
            empty_idx: Vec::new(),
        }
    }

    pub fn get_storage(&self) -> &Vec<Option<ElementType<F>>> {
        &self.storage
    }

    pub fn get_mut_storage(&mut self) -> &mut Vec<Option<ElementType<F>>> {
        &mut self.storage
    }

    pub fn get_permanent_storage(&self) -> &Vec<PermanentTensor<F>> {
        &self.permanent_storage
    }

    pub fn get_mut_permanent_storage(&mut self) -> &mut Vec<PermanentTensor<F>> {
        &mut self.permanent_storage
    }

    pub fn get_empty_idx(&self) -> &Vec<usize> {
        &self.empty_idx
    }

    pub fn get_mut_empty_idx(&mut self) -> &mut Vec<usize> {
        &mut self.empty_idx
    }

    pub fn push_permanent_tensor(&mut self, array: Array<F>, grad: Array<F>) -> StorageType {
        let idx = self.permanent_storage.len();
        self.permanent_storage.push(PermanentTensor { array, grad });
        StorageType::Permanent(idx)
    }

    pub fn push(&mut self, element: ElementType<F>) -> Result<StorageType, PzeudoErr> {
        let idx = if let Some(idx) = self.empty_idx.pop() {
            if self.storage[idx].is_some() {
                return Err(PzeudoErr::StoragePushErr(format!(
                    "ArrayStorage::push. The problem occurs because the index {idx} obtained from empty_idx points to an element that still has a value."
                )));
            }

            self.storage[idx].replace(element);
            idx
        } else {
            self.storage.push(Some(element));
            self.storage.len() - 1
        };

        Ok(StorageType::Storage(idx))
    }

    pub fn remove(&mut self, idx: usize) -> Result<(), PzeudoErr> {
        if self.storage[idx].is_none() {
            return Err(PzeudoErr::StorageRemoveErr(format!(
                "ArrayStorage::remove. index {idx} points to an element that has a value of None"
            )));
        }

        self.storage[idx].take();
        self.empty_idx.push(idx);

        Ok(())
    }

    pub fn get_element(&self, idx: StorageType) -> Result<GetElementOutput<'_, F>, PzeudoErr> {
        let data = match idx {
            StorageType::Permanent(idx) => {
                let data = self
                    .permanent_storage
                    .get(idx)
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get. index {idx} points to an invalid location on storage."
                    )))?;
                GetElementOutput::Permanent(data)
            }
            StorageType::Storage(idx) => {
                let data = self
                    .storage
                    .get(idx)
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get. index {idx} points to an invalid location on storage."
                    )))?
                    .as_ref()
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get. index {idx} points to elements that have the value None in storage."
                    )))?;
                GetElementOutput::Storage(data)
            }
        };

        Ok(data)
    }

    pub fn get_element_mut(
        &mut self,
        idx: StorageType,
    ) -> Result<GetElementMutOutput<'_, F>, PzeudoErr> {
        let data = match idx {
            StorageType::Permanent(idx) => {
                let data = self
                    .permanent_storage
                    .get_mut(idx)
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get. index {idx} points to an invalid location on storage."
                    )))?;
                GetElementMutOutput::Permanent(data)
            }
            StorageType::Storage(idx) => {
                let data = self
                    .storage
                    .get_mut(idx)
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get. index {idx} points to an invalid location on storage."
                    )))?
                    .as_mut()
                    .ok_or(PzeudoErr::StorageGetErr(format!(
                        "ArrayStorage::get. index {idx} points to elements that have the value None in storage."
                    )))?;
                GetElementMutOutput::Storage(data)
            }
        };

        Ok(data)
    }

    pub fn get_as_array_ref<T>(
        &self,
        idx: StorageType,
        arr_contiguous_type: ContiguousType,
    ) -> Result<ArrayRef<'_, F, T>, PzeudoErr> {
        match idx {
            StorageType::Permanent(idx) => {
                let permanent = self.permanent_storage
                    .get(idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. Index of type permanent_storage points to {idx} which is an invalid location in storage"
                    )))?;

                let array = match arr_contiguous_type {
                    ContiguousType::Arr => &permanent.array,
                    ContiguousType::Grad => &permanent.grad,
                };

                return Ok(ArrayRef {
                    data: &array.data,
                    offset: array.offset,
                    shape: &array.shape,
                    stride: &array.stride,
                    _array_type: Default::default(),
                });
            }
            StorageType::Storage(idx) => {
                let element = self
                    .storage
                    .get(idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. index {idx} points to an invalid location on storage."
                    )))?
                    .as_ref()
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. index {idx} points to elements that have the value None in storage."
                    )))?;

                match element {
                    ElementType::Contiguous(array, contiguous_type) => {
                        storage_contiguous_type_check(&arr_contiguous_type, contiguous_type)
                            .map_err(|_| PzeudoErr::StorageGetAsArrayRefErr(format!(
                                "ArrayStorage::get_as_array_ref. Cannot retrieve array at element with index {idx} because the type being searched is {arr_contiguous_type:?} but the element is of type {contiguous_type:?}."
                            )))?;

                        return Ok(ArrayRef {
                            data: &array.data,
                            offset: array.offset,
                            shape: &array.shape,
                            stride: &array.stride,
                            _array_type: Default::default(),
                        });
                    }
                    ElementType::View(p_idx, metadata) => match p_idx {
                        StorageType::Permanent(p_idx) => {
                            let permanent = self.permanent_storage
                                    .get(*p_idx)
                                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                                        "ArrayStorage::get_as_array_ref. index {idx} points to a view that has index {p_idx} that points to permanent_storage, but index {p_idx} is not a valid index."
                                    )))?;

                            let array = match arr_contiguous_type {
                                ContiguousType::Arr => &permanent.array,
                                ContiguousType::Grad => &permanent.grad,
                            };

                            Ok(ArrayRef {
                                data: &array.data,
                                offset: metadata.offset,
                                shape: &metadata.shape,
                                stride: &metadata.stride,
                                _array_type: Default::default(),
                            })
                        }
                        StorageType::Storage(p_idx) => {
                            let element = self
                                .storage
                                .get(*p_idx)
                                .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                                    "ArrayStorage::get_as_array_ref. index {idx} points to a view that has index {p_idx} that points to storage, but index {p_idx} is not a valid index."
                                )))?
                                .as_ref()
                                .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                                    "ArrayStorage::get_as_array_ref. index {idx} points to a view that has index {p_idx} that points to storage, but index {p_idx} points to an element with the value None."
                                )))?;

                            match element {
                                ElementType::View(_, _) => {
                                    return Err(PzeudoErr::StorageGetAsArrayRefErr(format!(
                                        "ArrayStorage::get_as_array_ref. index {idx} points to the View element that has index {p_idx} which points to the element that has value View Also, View pointing to View is prohibited"
                                    )));
                                }
                                ElementType::Contiguous(array, contiguous_type) => {
                                    storage_contiguous_type_check(&arr_contiguous_type, contiguous_type)
                                        .map_err(|_| PzeudoErr::StorageGetAsArrayRefErr(format!(
                                            "ArrayStorage::get_as_array_ref. Cannot retrieve array at element with index {idx} because the type being searched is {arr_contiguous_type:?} but the element is of type {contiguous_type:?}."
                                        )))?;

                                    return Ok(ArrayRef {
                                        data: &array.data,
                                        offset: metadata.offset,
                                        shape: &metadata.shape,
                                        stride: &metadata.stride,
                                        _array_type: Default::default(),
                                    });
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    pub fn get_as_array_ref_mut(
        &mut self,
        idx: StorageType,
        arr_contiguous_type: ContiguousType,
    ) -> Result<ArrayRefMut<'_, F, Contiguous>, PzeudoErr> {
        match idx {
            StorageType::Permanent(idx) => {
                let permanent = self.permanent_storage
                    .get_mut(idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                        "ArrayStorage::get_as_array_ref_mut. Index of type permanent_storage points to {idx} which is an invalid location in storage"
                    )))?;

                let array = match arr_contiguous_type {
                    ContiguousType::Arr => &mut permanent.array,
                    ContiguousType::Grad => &mut permanent.grad,
                };

                return Ok(ArrayRefMut {
                    data: &mut array.data,
                    offset: array.offset,
                    shape: &array.shape,
                    stride: &array.stride,
                    _array_type: Default::default(),
                });
            }
            StorageType::Storage(idx) => {
                let element = self
                    .storage
                    .get_mut(idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                        "ArrayStorage::get_as_array_ref_mut. index {idx} points to an invalid location on storage."
                    )))?
                    .as_mut()
                    .ok_or(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                        "ArrayStorage::get_as_array_ref_mut. index {idx} points to elements that have the value None in storage."
                    )))?;

                match element {
                    ElementType::Contiguous(array, contiguous_type) => {
                        storage_contiguous_type_check(&arr_contiguous_type, contiguous_type)
                            .map_err(|_| PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                                "ArrayStorage::get_as_array_ref_mut. Cannot retrieve array at element with index {idx} because the type being searched is {arr_contiguous_type:?} but the element is of type {contiguous_type:?}."
                            )))?;

                        return Ok(ArrayRefMut {
                            data: &mut array.data,
                            offset: array.offset,
                            shape: &array.shape,
                            stride: &array.stride,
                            _array_type: Default::default(),
                        });
                    }
                    ElementType::View(_, _) => {
                        return Err(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                            "ArrayStorage::get_as_array_ref_mut. The index {idx} points to the View element, the View element cannot be changed (mut)"
                        )));
                    }
                }
            }
        }
    }

    pub fn clear_storage(&mut self) {
        self.storage.clear();
        self.empty_idx.clear();
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
