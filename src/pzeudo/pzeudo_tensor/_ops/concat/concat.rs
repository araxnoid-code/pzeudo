use std::ops::AddAssign;

pub use crate::prelude::*;

pub trait ConcatVector<F, T, G> {
    fn _iter(&self) -> std::slice::Iter<'_, Tensor<F, T, G>>;
    fn _len(&self) -> usize;
    fn _get(&self, idx: usize) -> Option<&Tensor<F, T, G>>;

    fn tensor_concat<ReqGrad>(
        self,
        axis: usize,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
        Self: Sized,
        F: Copy,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        let vector_len = self._len();
        if vector_len == 0 {
            return Err(PzeudoErr::OpsErr(format!("ConcatVector. Empty Vector")));
        }

        let first_tensor = self._get(0).unwrap();
        let first_shape = &first_tensor.shape;
        let mut first_storage = first_tensor.storage.borrow_mut();

        let mut out_shape = first_shape.to_vec();
        out_shape[axis] = 0;

        let mut check_shape = Vec::with_capacity(first_shape.len());
        let mut acc = 0;
        for v_idx in 0..vector_len {
            let tensor = self._get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
                "ConcatVector. The index {} in the vector index points to an invalid location.",
                v_idx
            )))?;

            check_shape.extend_from_slice(&tensor.shape);
            check_shape[axis] = 0;
            if check_shape != out_shape {
                return Err(PzeudoErr::OpsErr(format!(
                    "ConcatVector. terdeteksi shape yang tidak sama diluar axis"
                )));
            }
            check_shape.clear();

            acc += tensor.shape[axis];
        }
        out_shape[axis] = acc;

        let outter_len = first_shape[..axis].iter().product::<usize>();
        let mut vec = Vec::with_capacity(out_shape.iter().product::<usize>());
        for o_idx in 0..outter_len {
            for v_idx in 0..vector_len {
                let tensor = self._get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
                    "ConcatVector. The index {} in the vector index points to an invalid location.",
                    v_idx
                )))?;

                let array = first_storage
                    .get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

                if axis >= array.shape.len() {
                    return Err(PzeudoErr::OpsErr(format!(
                        "ConcatVector. axis {} is greater than the tensor dimension, which is only {}",
                        axis,
                        first_shape.len()
                    )));
                }

                let idx_len = array.shape[axis..].iter().product::<usize>();
                for idx in 0..idx_len {
                    let offset = idx + idx_len * o_idx;

                    vec.push(array.linear_index(offset)?);
                }
            }
        }

        let grad_idx = requires_grad.into_zeros_grad_storage(&out_shape, &mut first_storage)?;
        let array = Array::new(vec, 0, shape_to_stride(&out_shape), out_shape.clone());
        let array_idx = first_storage.push(ElementType::Arr(array))?;

        let grad_idx_list = self
            ._iter()
            .map(|tensor| tensor.get_grad_idx())
            .collect::<Vec<Option<StorageType>>>();
        let record_label = RecordLabel::Concat(grad_idx_list, axis, grad_idx);
        let mut record = self._get(0).unwrap().record.borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            out_shape,
            record_status,
            self._get(0).unwrap().record.clone(),
            self._get(0).unwrap().storage.clone(),
        );

        Ok(tensor)
    }
}

pub trait ConcatVectorRef<F, T, G> {
    fn _iter(&self) -> std::slice::Iter<'_, &Tensor<F, T, G>>;
    fn _len(&self) -> usize;
    fn _get(&self, idx: usize) -> Option<&&Tensor<F, T, G>>;

    fn tensor_concat<ReqGrad>(
        self,
        axis: usize,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
        Self: Sized,
        F: Copy,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        let vector_len = self._len();
        if vector_len == 0 {
            return Err(PzeudoErr::OpsErr(format!("ConcatVector. Empty Vector")));
        }

        let first_tensor = self._get(0).unwrap();
        let first_shape = &first_tensor.shape;
        let mut first_storage = first_tensor.storage.borrow_mut();

        let mut out_shape = first_shape.to_vec();
        out_shape[axis] = 0;

        let mut check_shape = Vec::with_capacity(first_shape.len());
        let mut acc = 0;
        for v_idx in 0..vector_len {
            let tensor = self._get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
                "ConcatVector. The index {} in the vector index points to an invalid location.",
                v_idx
            )))?;

            check_shape.extend_from_slice(&tensor.shape);
            check_shape[axis] = 0;
            if check_shape != out_shape {
                return Err(PzeudoErr::OpsErr(format!(
                    "ConcatVector. A dissimilar shape was detected off-axis."
                )));
            }
            check_shape.clear();

            acc += tensor.shape[axis];
        }
        out_shape[axis] = acc;

        let outter_len = first_shape[..axis].iter().product::<usize>();
        let mut vec = Vec::with_capacity(out_shape.iter().product::<usize>());
        for o_idx in 0..outter_len {
            for v_idx in 0..vector_len {
                let tensor = self._get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
                    "ConcatVector. The index {} in the vector index points to an invalid location.",
                    v_idx
                )))?;

                let array = first_storage
                    .get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

                if axis >= array.shape.len() {
                    return Err(PzeudoErr::OpsErr(format!(
                        "ConcatVector. axis {} is greater than the tensor dimension, which is only {}",
                        axis,
                        first_shape.len()
                    )));
                }

                let idx_len = array.shape[axis..].iter().product::<usize>();
                for idx in 0..idx_len {
                    let offset = idx + idx_len * o_idx;

                    vec.push(array.linear_index(offset)?);
                }
            }
        }

        let grad_idx = requires_grad.into_zeros_grad_storage(&out_shape, &mut first_storage)?;
        let array = Array::new(vec, 0, shape_to_stride(&out_shape), out_shape.clone());
        let array_idx = first_storage.push(ElementType::Arr(array))?;

        let grad_idx_list = self
            ._iter()
            .map(|tensor| tensor.get_grad_idx())
            .collect::<Vec<Option<StorageType>>>();
        let record_label = RecordLabel::Concat(grad_idx_list, axis, grad_idx);
        let mut record = self._get(0).unwrap().record.borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            out_shape,
            record_status,
            self._get(0).unwrap().record.clone(),
            self._get(0).unwrap().storage.clone(),
        );

        Ok(tensor)
    }
}

pub fn concat_backward<F>(
    list_arr_grad: &[StorageType],
    axis: usize,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: AddAssign + Copy,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        let grad = storage.take_grad(grad_idx)?;
        let grad_ref = grad.to_array_ref::<Contiguous>();

        let first_array =
            storage.get_as_array_ref::<View>(list_arr_grad[0], ContiguousType::Grad)?;
        let outter_len = first_array.shape[..axis].iter().product::<usize>();

        for o_idx in 0..outter_len {
            for v_idx in 0..list_arr_grad.len() {
                if is_no_grad_or_time_not_match_or_no_update(list_arr_grad[v_idx], storage)? {
                    continue;
                };

                let mut array_grad = storage
                    .get_as_array_ref_mut::<View>(list_arr_grad[v_idx], ContiguousType::Grad)?;

                let idx_len_0 = array_grad.shape[axis..].iter().product::<usize>();
                let idx_len_1 = grad_ref.shape[axis..].iter().product::<usize>();
                for idx in 0..idx_len_0 {
                    let idx_0 = idx + o_idx * idx_len_0;
                    let idx_1 = idx + v_idx * idx_len_0 + o_idx * idx_len_1;

                    *array_grad.linear_index_mut(idx_0)? += grad_ref.linear_index(idx_1)?;
                }
            }
        }

        storage.replace_grad(grad_idx, grad)?;
    }

    Ok(())
}
