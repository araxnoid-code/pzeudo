pub use crate::prelude::*;

pub trait ConcatVector<F, T> {
    fn concat<ReqGrad>(
        self,
        axis: usize,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
        Self: Sized;
}

impl<F, T, G> ConcatVector<F, T> for Vec<Tensor<F, T, G>>
where
    F: Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    fn concat<ReqGrad>(
        self,
        axis: usize,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
    {
        let vector_len = self.len();
        if vector_len == 0 {
            return Err(PzeudoErr::OpsErr(format!("ConcatVector. Empty Vector")));
        }

        let first_shape = &self[0].shape;
        let mut first_storage = self[0].storage.borrow_mut();

        let mut out_shape = first_shape.to_vec();
        out_shape[axis] = 0;
        for v_idx in 0..vector_len {
            let tensor = self.get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
                "ConcatVector. The index {} in the vector index points to an invalid location.",
                v_idx
            )))?;

            out_shape[axis] += tensor.shape[axis];
        }

        let outter_len = first_shape[..axis].iter().product::<usize>();
        let mut vec = Vec::with_capacity(out_shape.iter().product::<usize>());
        for o_idx in 0..outter_len {
            for v_idx in 0..vector_len {
                let tensor = self.get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
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
            .iter()
            .map(|tensor| tensor.get_grad_idx())
            .collect::<Vec<Option<StorageType>>>();
        let record_label = RecordLabel::Concat(grad_idx_list, axis, grad_idx);
        let mut record = self[0].record.borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            out_shape,
            record_status,
            self[0].record.clone(),
            self[0].storage.clone(),
        );

        Ok(tensor)
    }
}
