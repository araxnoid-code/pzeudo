use crate::prelude::*;

impl<F> ArrayStorage<F> {
    pub fn get_as_array_ref<T>(
        &self,
        element_type: StorageType,
        contiguous_type: ContiguousType,
    ) -> Result<ArrayRef<'_, F, T>, PzeudoErr> {
        match element_type {
            StorageType::Permanent(permanent_idx) => {
                let permanent = self.permanent_storage
                    .get(permanent_idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. Index of type permanent_storage points to {permanent_idx} which is an invalid location in storage"
                    )))?;

                let array = match contiguous_type {
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
            StorageType::Arr(array_idx) => match contiguous_type {
                ContiguousType::Arr => {
                    let array = self.arr_storage.get_arr(array_idx)?;
                    return Ok(ArrayRef {
                        data: &array.data,
                        offset: array.offset,
                        shape: &array.shape,
                        stride: &array.stride,
                        _array_type: Default::default(),
                    });
                }

                ContiguousType::Grad => {
                    let grad = self.grad_storage.get_grad(array_idx)?;
                    return Ok(ArrayRef {
                        data: &grad.data,
                        offset: grad.offset,
                        shape: &grad.shape,
                        stride: &grad.stride,
                        _array_type: Default::default(),
                    });
                }
            },
            StorageType::View(view_idx) => {
                let view = self.view_storage.get_metadata(view_idx)?;
                match view.arr_index {
                    ViewStorageType::Permanent(permanent_idx) => {
                        let permanent = self.permanent_storage
                            .get(permanent_idx)
                            .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                                "ArrayStorage::get_as_array_ref. index {view_idx} in view_storage has index y that points to permanent_storage. Index {permanent_idx} points to an invalid location."
                            )))?;

                        let array = match contiguous_type {
                            ContiguousType::Arr => &permanent.array,
                            ContiguousType::Grad => &permanent.grad,
                        };

                        return Ok(ArrayRef {
                            data: &array.data,
                            offset: view.offset,
                            shape: &view.shape,
                            stride: &view.stride,
                            _array_type: Default::default(),
                        });
                    }
                    ViewStorageType::Storage(storage_idx) => match contiguous_type {
                        ContiguousType::Arr => {
                            let array = self.arr_storage.get_arr(storage_idx)?;
                            return Ok(ArrayRef {
                                data: &array.data,
                                offset: view.offset,
                                shape: &view.shape,
                                stride: &view.stride,
                                _array_type: Default::default(),
                            });
                        }

                        ContiguousType::Grad => {
                            let grad = self.grad_storage.get_grad(storage_idx)?;
                            return Ok(ArrayRef {
                                data: &grad.data,
                                offset: view.offset,
                                shape: &view.shape,
                                stride: &view.stride,
                                _array_type: Default::default(),
                            });
                        }
                    },
                }
            }
        }
    }

    pub fn get_as_array_ref_mut<T>(
        &mut self,
        element_type: StorageType,
        contiguous_type: ContiguousType,
    ) -> Result<ArrayRefMut<'_, F, T>, PzeudoErr> {
        match element_type {
            StorageType::Permanent(permanent_idx) => {
                let permanent = self.permanent_storage
                    .get_mut(permanent_idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. Index of type permanent_storage points to {permanent_idx} which is an invalid location in storage"
                    )))?;

                let array = match contiguous_type {
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
            StorageType::Arr(array_idx) => match contiguous_type {
                ContiguousType::Arr => {
                    let array = self.arr_storage.get_arr_mut(array_idx)?;
                    return Ok(ArrayRefMut {
                        data: &mut array.data,
                        offset: array.offset,
                        shape: &array.shape,
                        stride: &array.stride,
                        _array_type: Default::default(),
                    });
                }

                ContiguousType::Grad => {
                    let grad = self.grad_storage.get_grad_mut(array_idx)?;
                    return Ok(ArrayRefMut {
                        data: &mut grad.data,
                        offset: grad.offset,
                        shape: &grad.shape,
                        stride: &grad.stride,
                        _array_type: Default::default(),
                    });
                }
            },
            StorageType::View(view_idx) => {
                let view = self.view_storage.get_metadata(view_idx)?;
                match view.arr_index {
                    ViewStorageType::Permanent(permanent_idx) => {
                        let permanent = self.permanent_storage
                            .get_mut(permanent_idx)
                            .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                                "ArrayStorage::get_as_array_ref. index {view_idx} in view_storage has index y that points to permanent_storage. Index {permanent_idx} points to an invalid location."
                            )))?;

                        let array = match contiguous_type {
                            ContiguousType::Arr => &mut permanent.array,
                            ContiguousType::Grad => &mut permanent.grad,
                        };

                        return Ok(ArrayRefMut {
                            data: &mut array.data,
                            offset: view.offset,
                            shape: &view.shape,
                            stride: &view.stride,
                            _array_type: Default::default(),
                        });
                    }
                    ViewStorageType::Storage(storage_idx) => match contiguous_type {
                        ContiguousType::Arr => {
                            let array = self.arr_storage.get_arr_mut(storage_idx)?;
                            return Ok(ArrayRefMut {
                                data: &mut array.data,
                                offset: view.offset,
                                shape: &view.shape,
                                stride: &view.stride,
                                _array_type: Default::default(),
                            });
                        }

                        ContiguousType::Grad => {
                            let grad = self.grad_storage.get_grad_mut(storage_idx)?;
                            return Ok(ArrayRefMut {
                                data: &mut grad.data,
                                offset: view.offset,
                                shape: &view.shape,
                                stride: &view.stride,
                                _array_type: Default::default(),
                            });
                        }
                    },
                }
            }
        }
    }
}
