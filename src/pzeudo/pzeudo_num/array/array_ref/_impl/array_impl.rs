use crate::prelude::*;

impl<F> ArrayTrait<F> for ArrayRef<'_, F, View> {
    fn get_metadata(&self) -> Metadata<'_, F> {
        Metadata {
            data: self.data,
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        }
    }

    fn linear_index(&self, index: usize) -> Result<F, PzeudoErr>
    where
        F: Copy,
    {
        if self.contiguous {
            Ok(self.data[index])
        } else {
            let Metadata {
                data,
                offset,
                stride,
                shape,
            } = self.get_metadata();

            let output_stride = shape_to_stride(shape);

            let mut idx = offset;
            for (shape, (stride, out_stride)) in
                shape.iter().zip(stride.iter().zip(output_stride.iter()))
            {
                let permute = (index / out_stride) % shape;
                idx += permute * stride;
            }

            if idx >= data.len() {
                return Err(PzeudoErr::OpsErr(format!(
                    "LinearIndexErr. linear_index. index points to {idx} but array only has length {:?}",
                    data.len()
                )));
            }

            Ok(data[idx])
        }
    }
}

impl<F> ArrayTrait<F> for ArrayRef<'_, F, Contiguous> {
    fn get_metadata(&self) -> Metadata<'_, F> {
        let metadata = Metadata {
            data: self.data,
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        };
        metadata
    }

    fn linear_index(&self, index: usize) -> Result<F, PzeudoErr>
    where
        F: Copy,
    {
        Ok(self.data[index])
    }
}

impl<F> ArrayTrait<F> for ArrayRefMut<'_, F, View> {
    fn get_metadata(&self) -> Metadata<'_, F> {
        Metadata {
            data: self.data,
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        }
    }

    fn linear_index(&self, index: usize) -> Result<F, PzeudoErr>
    where
        F: Copy,
    {
        if self.contiguous {
            Ok(self.data[index])
        } else {
            let Metadata {
                data,
                offset,
                stride,
                shape,
            } = self.get_metadata();

            let output_stride = shape_to_stride(shape);

            let mut idx = offset;
            for (shape, (stride, out_stride)) in
                shape.iter().zip(stride.iter().zip(output_stride.iter()))
            {
                let permute = (index / out_stride) % shape;
                idx += permute * stride;
            }

            if idx >= data.len() {
                return Err(PzeudoErr::OpsErr(format!(
                    "LinearIndexErr. linear_index. index points to {idx} but array only has length {:?}",
                    data.len()
                )));
            }

            Ok(data[idx])
        }
    }
}

impl<F> ArrayTrait<F> for ArrayRefMut<'_, F, Contiguous> {
    fn get_metadata(&self) -> Metadata<'_, F> {
        let metadata = Metadata {
            data: self.data,
            offset: self.offset,
            shape: self.shape,
            stride: self.stride,
        };
        metadata
    }

    fn linear_index(&self, index: usize) -> Result<F, PzeudoErr>
    where
        F: Copy,
    {
        Ok(self.data[index])
    }
}
