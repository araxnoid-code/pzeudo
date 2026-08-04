use crate::prelude::*;

impl<T, G> Tensor<f32, T, G> {
    pub fn matmul_2d<J, RhsG, OutGrad>(
        &self,
        rhs: &Tensor<f32, J, RhsG>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<f32, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<f32>,
        for<'a> ArrayRef<'a, f32, T>: OpsMatmul2DF32,
        for<'a> ArrayRef<'a, f32, J>: OpsMatmul2DF32,
    {
        let mut storage = self.storage.borrow_mut();
        let lhs_array: ArrayRef<'_, f32, T> =
            storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, f32, J> =
            storage.get_as_array_ref(rhs.array_idx, ContiguousType::Arr)?;

        let result = lhs_array.matmul_2d(&rhs_array)?;
        let shape = result.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(result))?;
        let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;

        let record_label = RecordLabel::Matmul2dF32(
            (self.array_idx, self.grad_idx),
            (rhs.array_idx, rhs.grad_idx),
            grad_idx,
        );
        self.record.borrow_mut().push(record_label);

        drop(storage);
        let tensor = Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        );

        Ok(tensor)
    }
}

impl<T, G> Tensor<f64, T, G> {
    pub fn matmul_2d<J, RhsG, OutGrad>(
        &self,
        rhs: &Tensor<f64, J, RhsG>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<f64, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<f64>,
        for<'a> ArrayRef<'a, f64, T>: OpsMatmul2DF64,
        for<'a> ArrayRef<'a, f64, J>: OpsMatmul2DF64,
    {
        let mut storage = self.storage.borrow_mut();
        let lhs_array: ArrayRef<'_, f64, T> =
            storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, f64, J> =
            storage.get_as_array_ref(rhs.get_array_idx(), ContiguousType::Arr)?;

        let result = lhs_array.matmul_2d(&rhs_array)?;
        let shape = result.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(result))?;
        let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;

        let record_label = RecordLabel::Matmul2dF64(
            (self.array_idx, self.grad_idx),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );
        self.record.borrow_mut().push(record_label);

        drop(storage);
        let tensor = Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        );

        Ok(tensor)
    }
}

pub fn matmul_2d_f32_backward(
    lhs_idx: StorageType,
    lhs_gradient_idx: Option<StorageType>,
    rhs_idx: StorageType,
    rhs_gradient_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<f32>,
) -> Result<(), PzeudoErr> {
    if let Some(gradient_idx) = gradient_idx {
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }

        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_gradient_idx {
            if !check_no_grad_or_time_not_match(lhs_grad_idx, storage)? {
                let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
                let gradient = gradient.matmul_2d(&rhs_value.t())?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&gradient)?;
            }
        }

        let gradient: ArrayRef<'_, f32, Contiguous> =
            storage.get_as_array_ref(gradient_idx, ContiguousType::Grad)?;

        if let Some(rhs_grad_idx) = rhs_gradient_idx {
            if !check_no_grad_or_time_not_match(rhs_grad_idx, storage)? {
                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let gradient = lhs_value.t().matmul_2d(&gradient)?;

                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_grad_idx, ContiguousType::Grad)?;
                rhs_gradient.add_assign(&gradient)?;
            }
        }
    }
    Ok(())
}

pub fn matmul_2d_f64_backward(
    lhs_idx: StorageType,
    lhs_gradient_idx: Option<StorageType>,
    rhs_idx: StorageType,
    rhs_gradient_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<f64>,
) -> Result<(), PzeudoErr> {
    if let Some(gradient_idx) = gradient_idx {
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_gradient_idx {
            if !check_no_grad_or_time_not_match(lhs_grad_idx, storage)? {
                let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
                let gradient = gradient.matmul_2d(&rhs_value.t())?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&gradient)?;
            }
        }

        let gradient: ArrayRef<'_, f64, Contiguous> =
            storage.get_as_array_ref(gradient_idx, ContiguousType::Grad)?;
        if let Some(rhs_grad_idx) = rhs_gradient_idx {
            if !check_no_grad_or_time_not_match(rhs_grad_idx, storage)? {
                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let gradient = lhs_value.t().matmul_2d(&gradient)?;

                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_grad_idx, ContiguousType::Grad)?;
                rhs_gradient.add_assign(&gradient)?;
            }
        }
    }
    Ok(())
}
