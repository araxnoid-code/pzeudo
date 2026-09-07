use std::ops::{AddAssign, Mul, Sub, SubAssign};

use num_traits::Float;

use crate::prelude::*;

/// # softmax_cross_entropy
/// ```md
/// softmax_cross_entropy(target, logit) = -∑target * ln(e^logit / (epsilon + ∑e^logit))
/// epsilon = 1e-7
/// ```
/// - The result of mse will be summed into a scalar (since pzeudo does not yet support 0D tensors/scalars, it returns a 1D tensor containing a single value).
/// - The backward pass using softmax_cross_entropy_backward computes gradients only for the prediction.
pub fn softmax_cross_entropy<F, T, J, LhsGrad, RhsGrad, ReqGrad>(
    target: &Tensor<F, T, LhsGrad>,
    logit: &Tensor<F, J, RhsGrad>,
    axis: usize,
    requires_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    F: Float + AddAssign + SubAssign,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
{
    let mut storage = logit.get_storage().borrow_mut();

    let target_array =
        storage.get_as_array_ref::<T>(target.get_array_idx(), ContiguousType::Arr)?;
    let logit_array = storage.get_as_array_ref::<J>(logit.get_array_idx(), ContiguousType::Arr)?;

    if target_array.shape != logit_array.shape {
        return Err(PzeudoErr::LossErr(format!(
            "cross_entropy_loss. actual shape: {:?}, predicted shape: {:?}. The shape of both tensors must be the same",
            target_array.shape, logit_array.shape
        )));
    }

    let len = logit_array.shape.iter().product::<usize>();

    let exp_axis = logit_array.sum_axis_closure(&[axis], true, |_, x| Ok(x.exp()))?;
    let exp_axis_broadcasted = exp_axis.broadcast(logit.get_shape())?;

    let mut sum = F::zero();
    let mut softmax = if requires_grad.is_grad() {
        Some(Vec::with_capacity(len))
    } else {
        None
    };
    let epsilon = F::from(1e-7).ok_or(PzeudoErr::LossErr(format!(
        "softmax_cross_entropy. cannot cast data type epsilon"
    )))?;
    for i in 0..len {
        let s =
            logit_array.linear_index(i)?.exp() / (exp_axis_broadcasted.linear_index(i)? + epsilon);
        sum -= target_array.linear_index(i)? * s.ln();
        if let Some(softmax) = &mut softmax {
            softmax.push(s);
        }
    }

    let array_idx = storage.push(ElementType::Arr(Array::from_vector_with_shape(
        vec![sum],
        &[1],
    )?))?;
    let grad_idx = requires_grad.into_zeros_grad_storage(&[1], &mut storage)?;

    let record_idx = if let Some(softmax) = softmax {
        let record_label = RecordLabel::SoftmaxCrossEntropy(
            softmax,
            target.get_array_idx(),
            logit.get_grad_idx(),
            grad_idx,
        );
        let mut record = logit.get_record().borrow_mut();
        let record_idx = RecordStatus::Record(record.len());
        record.push(record_label);
        Some(record_idx)
    } else {
        None
    };

    let tensor = Tensor::_new(
        array_idx,
        grad_idx,
        vec![1],
        record_idx,
        logit.get_record().clone(),
        logit.get_storage().clone(),
    );

    Ok(tensor)
}

/// ```md
/// dsoftmax_cross_entropy(target, logit)/dlogit = (softmax(logit) - target) * gradient
/// ```
pub fn softmax_cross_entropy_backward<F>(
    softmax: &[F],
    target: StorageType,
    logit_grad_idx: Option<StorageType>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + Sub<Output = F> + AddAssign + Mul<Output = F>,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        }

        if let Some(logit_grad_idx) = logit_grad_idx {
            storage.set_grad_update(logit_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(logit_grad_idx, storage)? {
                return Ok(());
            }

            let grad_val = storage
                .get_as_array_ref::<Contiguous>(grad_idx, ContiguousType::Grad)?
                .data[0];

            let mut logit_grad_take = storage.take_grad(logit_grad_idx)?;
            let mut logit_grad = logit_grad_take.to_array_ref_mut::<View>();

            let target_array = storage.get_as_array_ref::<View>(target, ContiguousType::Arr)?;

            let len = target_array.shape.iter().product::<usize>();
            for i in 0..len {
                *logit_grad.linear_index_mut(i)? +=
                    (softmax[i] - target_array.linear_index(i)?) * grad_val;
            }

            storage.replace_grad(logit_grad_idx, logit_grad_take)?;
        }
    }
    Ok(())
}
