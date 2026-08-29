use crate::prelude::*;
use num_traits::{Float, One, Zero};
use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{AddAssign, DivAssign, Mul, MulAssign, SubAssign},
};

pub struct LayerNorm<F, ReqGrad> {
    gamma: Option<Tensor<F, Contiguous, ReqGrad>>,
    beta: Option<Tensor<F, Contiguous, ReqGrad>>,
}

impl<F, G> LayerNorm<F, G> {
    pub fn new(
        hidden: Option<usize>,
        model_builder: &mut ModelBuilder<F>,
        requires_grad: G,
    ) -> Result<LayerNorm<F, G>, PzeudoErr>
    where
        F: Clone + One + Zero,
        G: ReqGradTrait<F>,
    {
        let module = model_builder.get_module();
        let (gamma, beta) = if let Some(hidden) = hidden {
            let gamma = Tensor::param_ones(&[hidden], module, requires_grad)?;
            let beta = Tensor::param_zeros(&[hidden], module, requires_grad)?;
            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(LayerNorm { beta, gamma })
    }

    pub fn forward<T, TensorGrad, ReqGrad>(
        &self,
        tensor: &Tensor<F, T, TensorGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        F: AddAssign + DivAssign + Float + SubAssign + Debug + MulAssign,
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

        let gamma_broadcasted: Option<ArrayRef<'_, F, Contiguous>> =
            self.gamma.as_ref().map_or(Ok(None), |gamma| {
                Ok(Some(storage.get_as_array_ref::<Contiguous>(
                    gamma.get_array_idx(),
                    ContiguousType::Arr,
                )?))
            })?;

        let beta_broadcasted: Option<ArrayRef<'_, F, Contiguous>> =
            self.beta.as_ref().map_or(Ok(None), |beta| {
                Ok(Some(storage.get_as_array_ref::<Contiguous>(
                    beta.get_array_idx(),
                    ContiguousType::Arr,
                )?))
            })?;

        for i in 0..len {
            let mut y = (array_tensor.linear_index(i)? - avg_broadcast.linear_index(i)?)
                / (var_broadcast.linear_index(i)? + epsilon).sqrt();

            if let (Some(gamma), Some(beta)) = (&gamma_broadcasted, &beta_broadcasted) {
                y *= gamma.linear_index(i)?;
                y += beta.linear_index(i)?;
            }

            vec.push(y);
        }

        let norm = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            &vec,
            &tensor.shape,
        )?))?;
        let grad = requires_grad.into_zeros_grad_storage(&tensor.shape, &mut storage)?;

        let record_idx = if requires_grad.is_grad() {
            let mut record = tensor.get_record().borrow_mut();
            let record_idx = RecordStatus::Record(record.len());
            let record_label = RecordLabel::LayerNorm(
                norm,
                tensor.get_grad_idx(),
                var.data,
                self.gamma
                    .as_ref()
                    .map(|gamma| (gamma.get_array_idx(), gamma.get_grad_idx())),
                self.beta
                    .as_ref()
                    .map(|gamma| (gamma.get_array_idx(), gamma.get_grad_idx())),
                grad,
            );
            record.push(record_label);
            Some(record_idx)
        } else {
            None
        };

        let tensor = Tensor::_new(
            norm,
            grad,
            tensor.shape.to_vec(),
            record_idx,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        Ok(tensor)
    }
}

pub fn layer_norm_backward<F>(
    output_arr_idx: StorageType,
    array_grad_idx: Option<StorageType>,
    var: &[F],
    gamma: Option<(StorageType, Option<StorageType>)>,
    beta: Option<(StorageType, Option<StorageType>)>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Float + AddAssign + MulAssign + Sum<&'a F> + Display + Debug,
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

            let mut arr_grad = storage.take_grad(arr_grad_idx)?;
            let gamma_comp = gamma.map_or(Ok(None), |(array_idx, gamma_grad_idx)| {
                if let Some(gamma_grad_idx) = gamma_grad_idx {
                    return Ok(Some((storage.take_grad(gamma_grad_idx)?, array_idx)));
                }

                Ok(None)
            })?;
            let beta_comp = beta.map_or(Ok(None), |(array_idx, beta_grad_idx)| {
                if let Some(beta_grad_idx) = beta_grad_idx {
                    return Ok(Some((storage.take_grad(beta_grad_idx)?, array_idx)));
                }

                Ok(None)
            })?;

            let mut arr_grad_ref = arr_grad.to_array_ref_mut::<View>();
            let axis = arr_grad_ref.shape.len() - 1;

            let grad = storage.take_grad(grad_idx)?;
            let grad_ref = grad.to_array_ref::<Contiguous>();

            let out_array =
                storage.get_as_array_ref::<Contiguous>(output_arr_idx, ContiguousType::Arr)?;

            let mut s = arr_grad_ref.shape.to_vec();
            s[axis] = 1;
            let mut stride = shape_to_stride(&s);
            stride[axis] = 0;

            let var_arr: ArrayRef<'_, F, View> = ArrayRef {
                data: var,
                offset: 0,
                shape: &arr_grad_ref.shape,
                stride: &stride,
                _array_type: Default::default(),
            };

            let avg_grad_mul_arr =
                if let (Some((gamma_grad, gamma_idx)), Some((beta_grad, beta_idx))) =
                    (gamma_comp, beta_comp)
                {
                    panic!();
                    let gamma_array =
                        storage.get_as_array_ref::<Contiguous>(gamma_idx, ContiguousType::Arr)?;

                    let len = grad_ref.shape.iter().product::<usize>();
                    let mut vec = Vec::with_capacity(len);
                    let one = F::one();
                    for i in 0..len {
                        let y = (grad_ref.linear_index(i)? + one) * gamma_array.linear_index(i)?;
                        vec.push(y);
                    }

                    let array = Array {
                        data: vec,
                        offset: 0,
                        shape: grad_ref.shape.to_vec(),
                        stride: grad_ref.stride.to_vec(),
                    };
                    array.avg_axis(&[axis], true)?
                } else {
                    grad_ref.mul(&out_array)?.avg_axis(&[axis], true)?
                };

            let avg_grad_ref = grad_ref.avg_axis(&[axis], true)?;

            let avg_grad_mul_arr_broadcasted = avg_grad_mul_arr.broadcast(arr_grad_ref.shape)?;
            let avg_grad_ref_broadcasted = avg_grad_ref.broadcast(arr_grad_ref.shape)?;

            let len = arr_grad_ref.shape.iter().product::<usize>();
            let one = F::one();

            for i in 0..len {
                let std = (var_arr.linear_index(i)? + epsilon).sqrt();

                // 1
                let mut y = one / std * grad_ref.linear_index(i)?;

                // 2
                y += -avg_grad_ref_broadcasted.linear_index(i)? / std;

                // 3
                y += -out_array.linear_index(i)? * avg_grad_mul_arr_broadcasted.linear_index(i)?
                    / std;

                *arr_grad_ref.linear_index_mut(i)? += y;
            }

            storage.replace_grad(grad_idx, grad)?;
            storage.replace_grad(arr_grad_idx, arr_grad)?;
        }
    }

    Ok(())
}
