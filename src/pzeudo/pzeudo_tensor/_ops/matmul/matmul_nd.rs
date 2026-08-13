use crate::prelude::*;

impl<T, G> Tensor<f32, T, G> {
    pub fn matmul_nd<J, RhsGrad, OutGrad>(
        &self,
        rhs: &Tensor<f32, J, RhsGrad>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<f32, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<f32>,
        for<'a> ArrayRef<'a, f32, T>: OpsMatmulNDF32,
        for<'a> ArrayRef<'a, f32, J>: OpsMatmulNDF32,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let rhs_array = storage.get_as_array_ref::<J>(rhs.get_array_idx(), ContiguousType::Arr)?;

        let result = OpsMatmulNDF32::matmul_nd(&lhs_array, &rhs_array)?;
        let shape = result.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(result))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::MatmulNdF32(
            (self.get_array_idx(), self.get_grad_idx()),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );

        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.record.clone(),
            self.storage.clone(),
        ))
    }
}

impl<T, G> Tensor<f64, T, G> {
    pub fn matmul_nd<J, RhsGrad, OutGrad>(
        &self,
        rhs: &Tensor<f64, J, RhsGrad>,
        requires_grad: OutGrad,
    ) -> Result<Tensor<f64, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<f64>,
        for<'a> ArrayRef<'a, f64, T>: OpsMatmulNDF64,
        for<'a> ArrayRef<'a, f64, J>: OpsMatmulNDF64,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let rhs_array = storage.get_as_array_ref::<J>(rhs.get_array_idx(), ContiguousType::Arr)?;

        let result = OpsMatmulNDF64::matmul_nd(&lhs_array, &rhs_array)?;
        let shape = result.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(result))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::MatmulNdF64(
            (self.get_array_idx(), self.get_grad_idx()),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );

        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.record.clone(),
            self.storage.clone(),
        ))
    }
}

pub fn matmul_nd_f32_backward(
    lhs_idx: StorageType,
    lhs_gradient_idx: Option<StorageType>,
    rhs_idx: StorageType,
    rhs_gradient_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<f32>,
) -> Result<(), PzeudoErr> {
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();

        if let Some(lhs_gradient_idx) = lhs_gradient_idx {
            storage.set_grad_update(lhs_gradient_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_gradient_idx, storage)? {
                let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
                let dim = rhs_value.shape.len();
                let mut rhs_permute_idx = Vec::from_iter(0..dim);
                rhs_permute_idx[dim - 1] = dim - 2;
                rhs_permute_idx[dim - 2] = dim - 1;

                let rhs_permute = rhs_value.permute(&rhs_permute_idx)?;
                let grad = gradient_ref.matmul_nd(&rhs_permute)?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_gradient_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&grad)?;
            }
        }

        if let Some(rhs_gradient_idx) = rhs_gradient_idx {
            storage.set_grad_update(rhs_gradient_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(rhs_gradient_idx, storage)? {
                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let dim = lhs_value.shape.len();
                let mut lhs_permute_idx = Vec::from_iter(0..dim);
                lhs_permute_idx[dim - 1] = dim - 2;
                lhs_permute_idx[dim - 2] = dim - 1;

                let lhs_permute = lhs_value.permute(&lhs_permute_idx)?;
                let grad = lhs_permute.matmul_nd(&gradient_ref)?;

                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_gradient_idx, ContiguousType::Grad)?;
                rhs_gradient.add_assign(&grad)?;
            }
        }
        storage.replace_grad(gradient_idx, gradient)?;
    }
    Ok(())
}

pub fn matmul_nd_f64_backward(
    lhs_idx: StorageType,
    lhs_gradient_idx: Option<StorageType>,
    rhs_idx: StorageType,
    rhs_gradient_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<f64>,
) -> Result<(), PzeudoErr> {
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();

        if let Some(lhs_gradient_idx) = lhs_gradient_idx {
            storage.set_grad_update(lhs_gradient_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_gradient_idx, storage)? {
                let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
                let dim = rhs_value.shape.len();
                let mut rhs_permute_idx = Vec::from_iter(0..dim);
                rhs_permute_idx[dim - 1] = dim - 2;
                rhs_permute_idx[dim - 2] = dim - 1;

                let rhs_permute = rhs_value.permute(&rhs_permute_idx)?;
                let grad = gradient_ref.matmul_nd(&rhs_permute)?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_gradient_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&grad)?;
            }
        }

        if let Some(rhs_gradient_idx) = rhs_gradient_idx {
            storage.set_grad_update(rhs_gradient_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(rhs_gradient_idx, storage)? {
                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let dim = lhs_value.shape.len();
                let mut lhs_permute_idx = Vec::from_iter(0..dim);
                lhs_permute_idx[dim - 1] = dim - 2;
                lhs_permute_idx[dim - 2] = dim - 1;

                let lhs_permute = lhs_value.permute(&lhs_permute_idx)?;
                let grad = lhs_permute.matmul_nd(&gradient_ref)?;

                let mut rhs_gradient =
                    storage.get_as_array_ref_mut::<View>(rhs_gradient_idx, ContiguousType::Grad)?;
                rhs_gradient.add_assign(&grad)?;
            }
        }
        storage.replace_grad(gradient_idx, gradient)?;
    }
    Ok(())
}
