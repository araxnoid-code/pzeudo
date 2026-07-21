use std::format;

use crate::prelude::*;

pub struct ArrayStorage<F> {
    update_able_tensor_storage: Vec<UpdateAbleTensor<F>>,
    storage: Vec<Option<ElementType<F>>>,
    empty_idx: Vec<usize>,
}

impl<F> ArrayStorage<F> {
    pub fn new(capacity: Option<usize>) -> ArrayStorage<F> {
        let len = capacity.unwrap_or(1);
        Self {
            update_able_tensor_storage: Vec::new(),
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

    pub fn get_update_able_storage(&self) -> &Vec<UpdateAbleTensor<F>> {
        &self.update_able_tensor_storage
    }

    pub fn get_mut_update_able_storage(&mut self) -> &mut Vec<UpdateAbleTensor<F>> {
        &mut self.update_able_tensor_storage
    }

    pub fn get_empty_idx(&self) -> &Vec<usize> {
        &self.empty_idx
    }

    pub fn get_mut_empty_idx(&mut self) -> &mut Vec<usize> {
        &mut self.empty_idx
    }

    pub fn push_update_able_tensor(
        &mut self,
        array: Array<F>,
        grad: Array<F>,
    ) -> Result<usize, PzeudoErr> {
        let idx = self.update_able_tensor_storage.len();
        self.update_able_tensor_storage
            .push(UpdateAbleTensor { array, grad });
        self.push(ElementType::UpdateableTensor(idx))
    }

    pub fn push(&mut self, element: ElementType<F>) -> Result<usize, PzeudoErr> {
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

        Ok(idx)
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

    pub fn get_element(&self, idx: usize) -> Result<&ElementType<F>, PzeudoErr> {
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

        Ok(data)
    }

    pub fn get_element_mut(&mut self, idx: usize) -> Result<&mut ElementType<F>, PzeudoErr> {
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

        Ok(data)
    }

    pub fn get_as_array_ref<T>(&self, idx: usize) -> Result<ArrayRef<'_, F, T>, PzeudoErr> {
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
            ElementType::Contiguous(array) => Ok(ArrayRef {
                data: &array.data,
                offset: array.offset,
                shape: &array.shape,
                stride: &array.stride,
                _array_type: Default::default(),
            }),

            ElementType::UpdateableTensor(p_idx) => {
                let permanent_array = &self.update_able_tensor_storage
                    .get(*p_idx)
                    .ok_or(
                        PzeudoErr::StorageGetAsArrayRefErr(
                            format!(
                                "ArrayStorage::get_as_array_ref_mut. index {idx} points to update_able_tensor_storage index {p_idx}, but index {p_idx} points to an invalid location in update_able_tensor_storage."
                            )))?.array;
                Ok(ArrayRef {
                    data: &permanent_array.data,
                    offset: permanent_array.offset,
                    shape: &permanent_array.shape,
                    stride: &permanent_array.stride,
                    _array_type: Default::default(),
                })
            }

            ElementType::View(array_idx, metadata) => {
                let element = self
                    .storage
                    .get(*array_idx)
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. index {idx} points to a View element that has index {array_idx} that points to an invalid location in storage."
                    )))?
                    .as_ref()
                    .ok_or(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. index {idx} goes to the View element that has index {array_idx} which goes to the element that has the value None"
                    )))?;

                match element {
                    ElementType::View(_, _) => Err(PzeudoErr::StorageGetAsArrayRefErr(format!(
                        "ArrayStorage::get_as_array_ref. index {idx} points to the View element that has index {array_idx} which points to the element that has value View Also, View pointing to View is prohibited"
                    ))),

                    ElementType::UpdateableTensor(p_idx) => {
                        let permanent_array = &self.update_able_tensor_storage
                            .get(*p_idx)
                            .ok_or(
                                PzeudoErr::StorageGetAsArrayRefErr(
                                    format!(
                                        "ArrayStorage::get_as_array_ref_mut. index {idx} points to update_able_tensor_storage index {p_idx}, but index {p_idx} points to an invalid location in update_able_tensor_storage."
                                    )))?.array;
                        Ok(ArrayRef {
                            data: &permanent_array.data,
                            offset: metadata.offset,
                            shape: &metadata.shape,
                            stride: &metadata.stride,
                            _array_type: Default::default(),
                        })
                    }

                    ElementType::Contiguous(array) => Ok(ArrayRef {
                        data: &array.data,
                        offset: metadata.offset,
                        shape: &metadata.shape,
                        stride: &metadata.stride,
                        _array_type: Default::default(),
                    }),
                }
            }
        }
    }

    pub fn get_as_array_ref_mut(
        &mut self,
        idx: usize,
    ) -> Result<ArrayRefMut<'_, F, Contiguous>, PzeudoErr> {
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
            ElementType::Contiguous(array) => Ok(ArrayRefMut {
                data: &mut array.data,
                offset: array.offset,
                shape: &array.shape,
                stride: &array.stride,
                _array_type: Default::default(),
            }),

            ElementType::UpdateableTensor(p_idx) => {
                let permanent_array = &mut self.update_able_tensor_storage.get_mut(*p_idx).ok_or(
                    PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                        "ArrayStorage::get_as_array_ref_mut. index {idx} points to update_able_tensor_storage index {p_idx}, but index {p_idx} points to an invalid location in update_able_tensor_storage."
                    )),
                )?.array;

                Ok(ArrayRefMut {
                    data: &mut permanent_array.data,
                    offset: permanent_array.offset,
                    shape: &permanent_array.shape,
                    stride: &permanent_array.stride,
                    _array_type: Default::default(),
                })
            }

            ElementType::View(_, _) => Err(PzeudoErr::StorageGetAsArrayRefMutErr(format!(
                "ArrayStorage::get_as_array_ref_mut. The index {idx} points to the View element, the View element cannot be changed (mut)"
            ))),
        }
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
