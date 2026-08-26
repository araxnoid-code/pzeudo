use crate::prelude::*;
use num_traits::Float;
use std::ops::{AddAssign, DivAssign, Mul, MulAssign, SubAssign};

pub struct LayerNorm {}

impl LayerNorm {
    pub fn forward<F, T, G, ReqGrad>(
        &self,
        tensor: &Tensor<F, T, G>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        F: AddAssign + DivAssign + Float + SubAssign,
        for<'a> &'a F: Mul<&'a F, Output = F>,
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        let epsilon = F::from(1e-7).ok_or(PzeudoErr::LayerErr(format!(
            "LayerNorm::forward. Cannot cast to epsilon."
        )))?;
        let axis = tensor.shape.len() - 1;

        let mut storage = tensor.get_storage().borrow_mut();
        let array_tensor =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

        let (avg, var) = array_tensor.avg_and_var_axis(&[axis], true)?;

        let avg_broadcast = avg.broadcast(&tensor.shape)?;
        let var_broadcast = var.broadcast(&tensor.shape)?;

        let len = array_tensor.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let y = (array_tensor.linear_index(i)? - avg_broadcast.linear_index(i)?)
                / (var_broadcast.linear_index(i)? + epsilon).sqrt();
            vec.push(y);
        }

        let norm = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            &vec,
            &tensor.shape,
        )?))?;
        let grad = requires_grad.into_zeros_grad_storage(&tensor.shape, &mut storage)?;

        let tensor = Tensor::_new(
            norm,
            grad,
            tensor.shape.to_vec(),
            None,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        Ok(tensor)
    }
}

pub fn layer_norm_backward<F>(
    array_idx: StorageType,
    array_grad_idx: Option<StorageType>,
    avg: &[F],
    var: &[F],
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + AddAssign + MulAssign,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        }

        if let Some(arr_grad_idx) = array_grad_idx {
            storage.set_grad_update(arr_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(arr_grad_idx, storage)? {
                return Ok(());
            }
            let epsilon = F::from(1e-7).ok_or(PzeudoErr::BackwardErr(format!(
                "layer_norm_backwar. tidak dapat melakukan casting pada epsilon."
            )))?;

            let grad = storage.take_grad(grad_idx)?;
            let grad_ref = grad.to_array_ref::<Contiguous>();

            let mut arr_grad = storage.take_grad(arr_grad_idx)?;
            let mut arr_grad_ref = arr_grad.to_array_ref_mut::<View>();

            let n = F::from(arr_grad_ref.shape[arr_grad_ref.shape.len() - 1]).ok_or(
                PzeudoErr::BackwardErr(format!(
                    "layer_norm_backwar. tidak dapat melakukan casting pada n."
                )),
            )?;

            let mut stride = shape_to_stride(&arr_grad_ref.shape);
            stride[arr_grad_ref.shape.len() - 1] = 0;

            let array = storage.get_as_array_ref::<View>(array_idx, ContiguousType::Arr)?;

            let avg_arr: ArrayRef<'_, F, View> = ArrayRef {
                data: avg,
                offset: 0,
                shape: &arr_grad_ref.shape,
                stride: &stride,
                _array_type: Default::default(),
            };

            let var_arr: ArrayRef<'_, F, View> = ArrayRef {
                data: var,
                offset: 0,
                shape: &arr_grad_ref.shape,
                stride: &stride,
                _array_type: Default::default(),
            };

            let len = arr_grad_ref.shape.iter().product::<usize>();
            let two = F::one() + F::one();
            for i in 0..len {
                let mut y = F::one() / (var_arr.linear_index(i)? + epsilon).sqrt();
                y += -F::one() / ((var_arr.linear_index(i)? + epsilon).sqrt() * n);

                let a = array.linear_index(i)? - avg_arr.linear_index(i)?;
                let a = two * a * a;
                let b = var_arr.linear_index(i)? + epsilon;
                let b = two * n * b * b.sqrt();
                y += -a / b;
                y *= grad_ref.linear_index(i)?;

                *arr_grad_ref.linear_index_mut(i)? += y;
            }

            storage.replace_grad(grad_idx, grad)?;
            storage.replace_grad(arr_grad_idx, arr_grad)?;
        }
    }

    Ok(())
}
