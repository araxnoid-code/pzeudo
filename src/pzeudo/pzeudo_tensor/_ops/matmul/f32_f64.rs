use crate::prelude::*;

pub trait F32F64MatmulTensor<F> {
    fn matmul_2d<T, LhsGrad, J, RhsGrad, ReqGrad>(
        lhs: &Tensor<F, T, LhsGrad>,
        rhs: &Tensor<F, J, RhsGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>;
}

impl F32F64MatmulTensor<f32> for f32 {
    fn matmul_2d<T, LhsGrad, J, RhsGrad, ReqGrad>(
        lhs: &Tensor<f32, T, LhsGrad>,
        rhs: &Tensor<f32, J, RhsGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<f32, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<f32>,
        for<'a> ArrayRef<'a, f32, T>: ArrayTrait<f32>,
        for<'a> ArrayRef<'a, f32, J>: ArrayTrait<f32>,
    {
        Ok(lhs.matmul_2d(&rhs, requires_grad)?)
    }
}

impl F32F64MatmulTensor<f64> for f64 {
    fn matmul_2d<T, LhsGrad, J, RhsGrad, ReqGrad>(
        lhs: &Tensor<f64, T, LhsGrad>,
        rhs: &Tensor<f64, J, RhsGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<f64, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<f64>,
        for<'a> ArrayRef<'a, f64, T>: ArrayTrait<f64>,
        for<'a> ArrayRef<'a, f64, J>: ArrayTrait<f64>,
    {
        Ok(lhs.matmul_2d(&rhs, requires_grad)?)
    }
}
