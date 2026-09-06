use crate::prelude::*;
use num_traits::Zero;
use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

impl<F, T, G> Tensor<F, T, G> {
    pub fn add<J, RhsGrad, OutGrad>(
        &self,
        rhs: &Tensor<F, J, RhsGrad>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
        F: Copy + Add<Output = F> + Zero + Clone,
        for<'a> ArrayRef<'a, F, T>: OpsAdd<F> + OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, J>: OpsAdd<F> + OpsBroadcast<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let lhs_array: ArrayRef<'_, F, T> =
            storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, F, J> =
            storage.get_as_array_ref(rhs.array_idx, ContiguousType::Arr)?;

        let array = OpsAdd::add(&lhs_array, &rhs_array)?;
        let shape = array.shape.to_vec();

        let (lhs_broadcast, rhs_broadcast) = broadcast_detect(lhs_array.shape, rhs_array.shape);

        let array_idx = storage.push(ElementType::Arr(array))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Add(
            (self.array_idx, self.grad_idx, lhs_broadcast),
            (rhs.array_idx, rhs.grad_idx, rhs_broadcast),
            grad_idx,
        );

        let mut record = self.get_record().borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.record.clone(),
            self.storage.clone(),
        ))
    }

    pub fn add_scalar<OutGrad>(
        &self,
        lhs_scalar: F,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Copy + Add<Output = F>,
    {
        let mut storage = self.storage.borrow_mut();

        let arr = storage
            .get_as_array_ref::<T>(self.array_idx, ContiguousType::Arr)?
            .add_scalar(lhs_scalar)?;
        let shape = arr.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let mut record = self.get_record().borrow_mut();
        let record_label = RecordLabel::AddScalar(self.get_grad_idx(), grad_idx);
        let record_idx = RecordStatus::Record(record.len());
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            Some(record_idx),
            self.get_record().clone(),
            self.get_storage().clone(),
        );

        Ok(tensor)
    }

    pub fn scalar_add<OutGrad>(
        &self,
        rhs_scalar: F,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Copy + Add<Output = F>,
    {
        let mut storage = self.storage.borrow_mut();

        let arr = storage
            .get_as_array_ref::<T>(self.array_idx, ContiguousType::Arr)?
            .scalar_add(rhs_scalar)?;
        let shape = arr.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let mut record = self.get_record().borrow_mut();
        let record_label = RecordLabel::AddScalar(self.get_grad_idx(), grad_idx);
        let record_idx = RecordStatus::Record(record.len());
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            Some(record_idx),
            self.get_record().clone(),
            self.get_storage().clone(),
        );

        Ok(tensor)
    }
}

pub fn add_scalar_backward<F>(
    arr_grad_idx: Option<StorageType>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + AddAssign,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        if let Some(arr_grad_idx) = arr_grad_idx {
            storage.set_grad_update(arr_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(arr_grad_idx, storage)? {
                return Ok(());
            };

            let grad_take = storage.take_grad(grad_idx)?;
            let grad = grad_take.to_array_ref::<Contiguous>();

            let mut arr_grad =
                storage.get_as_array_ref_mut::<View>(grad_idx, ContiguousType::Grad)?;
            let len = arr_grad.shape.iter().product::<usize>();
            for i in 0..len {
                *arr_grad.linear_index_mut(i)? += grad.linear_index(i)?;
            }

            storage.replace_grad(grad_idx, grad_take)?;
        }
    }

    Ok(())
}

pub fn add_backward<F>(
    gradient_idx: Option<StorageType>,
    lhs_grad: Option<StorageType>,
    lhs_broadcast_dim: Option<&Vec<usize>>,
    rhs_grad: Option<StorageType>,
    rhs_broadcast_dim: Option<&Vec<usize>>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Clone + AddAssign + Copy + Zero + Sum<&'a F>,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        };
        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();

        if let Some(lhs_grad) = lhs_grad {
            storage.set_grad_update(lhs_grad, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad, storage)? {
                let mut lhs_gradient: ArrayRefMut<'_, F, View> =
                    storage.get_as_array_ref_mut(lhs_grad, ContiguousType::Grad)?;
                match lhs_broadcast_dim {
                    Some(dim) => {
                        let gradient = OpsSum::sum_axis(&gradient_ref, dim, true)?;
                        let to_shape = gradient.to_shape(lhs_gradient.shape)?;
                        lhs_gradient.add_assign(&to_shape)?
                    }
                    None => lhs_gradient.add_assign(&gradient_ref)?,
                }
            };
        }

        if let Some(rhs_grad) = rhs_grad {
            storage.set_grad_update(rhs_grad, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(rhs_grad, storage)? {
                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_grad, ContiguousType::Grad)?;

                match rhs_broadcast_dim {
                    Some(dim) => {
                        let gradient = gradient_ref.sum_axis(dim, true)?;
                        let to_shape = gradient.to_shape(rhs_gradient.shape)?;
                        rhs_gradient.add_assign(&to_shape)?;
                    }
                    None => rhs_gradient.add_assign(&gradient_ref)?,
                }
                storage.set_grad_update(rhs_grad, true)?;
            };
        }

        storage.replace_grad(gradient_idx, gradient)?;
    }

    Ok(())
}
