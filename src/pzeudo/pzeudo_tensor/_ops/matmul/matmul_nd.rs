use crate::prelude::*;

impl<T> Tensor<f32, T> {
    pub fn matmul_nd<J>(&self, rhs: &Tensor<f32, J>) -> Result<Tensor<f32, Contiguous>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f32, T>: OpsMatmulNDF32,
        for<'a> ArrayRef<'a, f32, J>: OpsMatmulNDF32,
    {
        let mut borrow_mut_storage = self.get_storage().borrow_mut();

        let lhs_array =
            borrow_mut_storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let rhs_array =
            borrow_mut_storage.get_as_array_ref::<J>(rhs.get_array_idx(), ContiguousType::Arr)?;

        let result = OpsMatmulNDF32::matmul_nd(&lhs_array, &rhs_array)?;
        let grad = Array::<f32>::zeros(&result.shape);

        let array_idx =
            borrow_mut_storage.push(ElementType::Contiguous(result, ContiguousType::Arr))?;
        let grad_idx =
            Some(borrow_mut_storage.push(ElementType::Contiguous(grad, ContiguousType::Grad))?);

        let record_label = RecordLabel::MatmulNdF32(
            (self.get_array_idx(), self.get_grad_idx()),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor {
            array_idx,
            grad_idx,
            record: self.get_record().clone(),
            storage: self.get_storage().clone(),
            _array_type: Default::default(),
        })
    }
}

impl<T> Tensor<f64, T> {
    pub fn matmul_nd<J>(&self, rhs: &Tensor<f64, J>) -> Result<Tensor<f64, Contiguous>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f64, T>: OpsMatmulNDF64,
        for<'a> ArrayRef<'a, f64, J>: OpsMatmulNDF64,
    {
        let mut borrow_mut_storage = self.get_storage().borrow_mut();

        let lhs_array =
            borrow_mut_storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;

        let rhs_array =
            borrow_mut_storage.get_as_array_ref::<J>(rhs.get_array_idx(), ContiguousType::Arr)?;

        let result = OpsMatmulNDF64::matmul_nd(&lhs_array, &rhs_array)?;
        let grad = Array::<f64>::zeros(&result.shape);

        let array_idx =
            borrow_mut_storage.push(ElementType::Contiguous(result, ContiguousType::Arr))?;
        let grad_idx =
            Some(borrow_mut_storage.push(ElementType::Contiguous(grad, ContiguousType::Grad))?);

        let record_label = RecordLabel::MatmulNdF64(
            (self.get_array_idx(), self.get_grad_idx()),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor {
            array_idx,
            grad_idx,
            record: self.get_record().clone(),
            storage: self.get_storage().clone(),
            _array_type: Default::default(),
        })
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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_gradient_idx) = lhs_gradient_idx {
            let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
            let dim = rhs_value.shape.len();
            let mut rhs_permute_idx = Vec::from_iter(0..dim);
            rhs_permute_idx[dim - 1] = dim - 2;
            rhs_permute_idx[dim - 2] = dim - 1;

            let rhs_permute = rhs_value.permute(&rhs_permute_idx)?;
            let grad = gradient.matmul_nd(&rhs_permute)?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut(lhs_gradient_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&grad)?;
        }

        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;
        if let Some(rhs_gradient_idx) = rhs_gradient_idx {
            let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
            let dim = lhs_value.shape.len();
            let mut lhs_permute_idx = Vec::from_iter(0..dim);
            lhs_permute_idx[dim - 1] = dim - 2;
            lhs_permute_idx[dim - 2] = dim - 1;

            let lhs_permute = lhs_value.permute(&lhs_permute_idx)?;
            let grad = lhs_permute.matmul_nd(&gradient)?;

            let mut rhs_gradient =
                storage.get_as_array_ref_mut(rhs_gradient_idx, ContiguousType::Grad)?;
            rhs_gradient.add_assign(&grad)?;
        }
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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_gradient_idx) = lhs_gradient_idx {
            let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
            let dim = rhs_value.shape.len();
            let mut rhs_permute_idx = Vec::from_iter(0..dim);
            rhs_permute_idx[dim - 1] = dim - 2;
            rhs_permute_idx[dim - 2] = dim - 1;

            let rhs_permute = rhs_value.permute(&rhs_permute_idx)?;
            let grad = gradient.matmul_nd(&rhs_permute)?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut(lhs_gradient_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&grad)?;
        }

        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;
        if let Some(rhs_gradient_idx) = rhs_gradient_idx {
            let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
            let dim = lhs_value.shape.len();
            let mut lhs_permute_idx = Vec::from_iter(0..dim);
            lhs_permute_idx[dim - 1] = dim - 2;
            lhs_permute_idx[dim - 2] = dim - 1;

            let lhs_permute = lhs_value.permute(&lhs_permute_idx)?;
            let grad = lhs_permute.matmul_nd(&gradient)?;

            let mut rhs_gradient =
                storage.get_as_array_ref_mut(rhs_gradient_idx, ContiguousType::Grad)?;
            rhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}
