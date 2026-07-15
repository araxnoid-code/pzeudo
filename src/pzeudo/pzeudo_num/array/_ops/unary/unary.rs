use num_traits::Float;

use crate::{Array, ArrayTrait, PzeudoNumErr};

pub trait OpsUnary<F>: ArrayTrait<F>
where
    F: Copy + Float,
{
    fn exp(&self) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.exp();
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }

    fn ln(&self) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.ln();
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }

    fn log(&self, scalar: F) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.log(scalar);
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }

    fn log2(&self) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.log2();
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }

    fn log10(&self) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.log10();
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }

    fn powi(&self, scalar: i32) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.powi(scalar);
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }

    fn powf(&self, scalar: F) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.powf(scalar);
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }
}
