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
            return Err(PzeudoErr::OpsErr(format!("ConcatVector. Vector kosong")));
        }

        let first_shape = &self[0].shape;
        let mut first_storage = self[0].storage.borrow_mut();
        if axis >= first_shape.len() {
            return Err(PzeudoErr::OpsErr(format!(
                "ConcatVector. axis {} melebihi dimensi tensor yang hanya berdimensi {}",
                axis,
                first_shape.len()
            )));
        }

        let outter_len = first_shape[..axis].iter().product::<usize>();
        let idx_len = first_shape[axis..].iter().product::<usize>();

        let mut out_shape = first_shape.to_vec();
        out_shape[axis] *= vector_len;

        let mut vec = Vec::with_capacity(out_shape.iter().product::<usize>());
        for o_idx in 0..outter_len {
            for v_idx in 0..vector_len {
                for idx in 0..idx_len {
                    let offset = idx + idx_len * o_idx;
                    let tensor = self.get(v_idx).ok_or(PzeudoErr::OpsErr(format!(
                        "ConcatVector. index {} pada vector index mengarah pada lokasi yang tidak valid", v_idx
                    )))?;

                    if first_shape != &tensor.shape {
                        return Err(PzeudoErr::OpsErr(format!(
                            "ConcatVector. shape pada tensor tidak sama",
                        )));
                    }

                    let array = first_storage
                        .get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
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
