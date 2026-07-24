use crate::prelude::*;

impl<T> Tensor<f32, T> {
    pub fn matmul_2d<J>(&self, rhs: &Tensor<f32, J>) -> Result<Tensor<f32, Contiguous>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f32, T>: OpsMatmul2DF32,
        for<'a> ArrayRef<'a, f32, J>: OpsMatmul2DF32,
    {
        let mut borrow_mut_storage = self.storage.borrow_mut();
        let lhs_array: ArrayRef<'_, f32, T> =
            borrow_mut_storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, f32, J> =
            borrow_mut_storage.get_as_array_ref(rhs.array_idx, ContiguousType::Arr)?;

        let result = lhs_array.matmul_2d(&rhs_array)?;
        let grad = Array::<f32>::zeros(&result.shape);
        let array_idx =
            borrow_mut_storage.push(ElementType::Contiguous(result, ContiguousType::Arr))?;
        let grad_idx =
            Some(borrow_mut_storage.push(ElementType::Contiguous(grad, ContiguousType::Grad))?);

        let record_label = RecordLabel::Matmul2dF32(
            (self.array_idx, self.grad_idx),
            (rhs.array_idx, rhs.grad_idx),
            grad_idx,
        );
        self.record.borrow_mut().push(record_label);

        drop(borrow_mut_storage);
        let tensor: Tensor<f32, Contiguous> = Tensor {
            array_idx,
            grad_idx,
            record: self.record.clone(),
            storage: self.storage.clone(),
            _array_type: Default::default(),
        };

        Ok(tensor)
    }
}

impl<T> Tensor<f64, T> {
    pub fn matmul_2d<J>(&self, rhs: &Tensor<f64, J>) -> Result<Tensor<f64, Contiguous>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f64, T>: OpsMatmul2DF64,
        for<'a> ArrayRef<'a, f64, J>: OpsMatmul2DF64,
    {
        let mut borrow_mut_storage = self.storage.borrow_mut();
        let lhs_array: ArrayRef<'_, f64, T> =
            borrow_mut_storage.get_as_array_ref(self.array_idx, ContiguousType::Arr)?;
        let rhs_array: ArrayRef<'_, f64, J> =
            borrow_mut_storage.get_as_array_ref(rhs.get_array_idx(), ContiguousType::Arr)?;

        let result = lhs_array.matmul_2d(&rhs_array)?;
        let grad = Array::<f64>::zeros(&result.shape);
        let array_idx =
            borrow_mut_storage.push(ElementType::Contiguous(result, ContiguousType::Arr))?;
        let grad_idx =
            Some(borrow_mut_storage.push(ElementType::Contiguous(grad, ContiguousType::Grad))?);

        let record_label = RecordLabel::Matmul2dF64(
            (self.array_idx, self.grad_idx),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );
        self.record.borrow_mut().push(record_label);

        drop(borrow_mut_storage);
        let tensor: Tensor<f64, Contiguous> = Tensor {
            array_idx,
            grad_idx,
            record: self.record.clone(),
            storage: self.storage.clone(),
            _array_type: Default::default(),
        };

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
        let gradient: ArrayRef<'_, f32, Contiguous> =
            storage.get_as_array_ref(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_gradient_idx {
            let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
            let gradient = gradient.matmul_2d(&rhs_value.t())?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut(lhs_grad_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&gradient)?;
        }

        let gradient: ArrayRef<'_, f32, Contiguous> =
            storage.get_as_array_ref(gradient_idx, ContiguousType::Grad)?;
        if let Some(rhs_grad_idx) = rhs_gradient_idx {
            let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
            let gradient = lhs_value.t().matmul_2d(&gradient)?;

            let mut rhs_gradient =
                storage.get_as_array_ref_mut(rhs_grad_idx, ContiguousType::Grad)?;
            rhs_gradient.add_assign(&gradient)?;
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
        let gradient: ArrayRef<'_, f64, Contiguous> =
            storage.get_as_array_ref(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_gradient_idx {
            let rhs_value = storage.get_as_array_ref::<View>(rhs_idx, ContiguousType::Arr)?;
            let gradient = gradient.matmul_2d(&rhs_value.t())?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut(lhs_grad_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&gradient)?;
        }

        let gradient: ArrayRef<'_, f64, Contiguous> =
            storage.get_as_array_ref(gradient_idx, ContiguousType::Grad)?;
        if let Some(rhs_grad_idx) = rhs_gradient_idx {
            let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
            let gradient = lhs_value.t().matmul_2d(&gradient)?;

            let mut rhs_gradient =
                storage.get_as_array_ref_mut(rhs_grad_idx, ContiguousType::Grad)?;
            rhs_gradient.add_assign(&gradient)?;
        }
    }
    Ok(())
}
