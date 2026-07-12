use crate::{
    Array, ArrayTrait, OpsMatmul2DF32, OpsMatmul2DF64,
    PzeudoNumErr::{self, MatmulNDErr},
    shape_to_stride,
};

pub trait OpsMatmulNDF32: OpsMatmul2DF32 {
    fn matmul_nd<Rhs>(&self, rhs: &Rhs) -> Result<Array<f32>, PzeudoNumErr>
    where
        Rhs: ArrayTrait<f32>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape.len() != rhs_metadata.shape.len() {
            return Err(MatmulNDErr(format!(
                "MatmulNDErr. OpsMatmulNDF32::matmul_nd. cannot perform matmul_nd on an array having shape {:?} and an array having shape {:?} due to unequal dimensions.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let lhs_shape = lhs_metadata.shape;
        let rhs_shape = rhs_metadata.shape;
        let dim = lhs_shape.len();

        if lhs_shape[..dim - 2] != rhs_shape[..dim - 2] {
            return Err(MatmulNDErr(format!(
                "MatmulNDErr. OpsMatmulNDF32::matmul_nd. cannot perform matmul_nd on an array having shape {:?} and an array having shape {:?} due to unequal batch sizes.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        if lhs_shape[dim - 1] != rhs_shape[dim - 2] {
            return Err(MatmulNDErr(format!(
                "MatmulNDErr. OpsMatmulNDF32::matmul_nd. cannot perform matmul_nd on an array having shape {:?} and an array having shape {:?} because it does not fulfill the shape '...×m×k * ...×k×n'.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let mut new_shape = lhs_shape[..dim - 2].to_vec();
        new_shape.push(lhs_shape[dim - 2]);
        new_shape.push(rhs_shape[dim - 1]);

        let mut vec = vec![0.; new_shape.iter().product::<usize>()];
        let matrix_count = lhs_shape[..dim - 2].iter().product::<usize>();
        for i in 0..matrix_count {
            let mut lhs_new_offset = lhs_metadata.offset;
            for (shape, stride) in lhs_shape[..dim - 3]
                .iter()
                .zip(lhs_metadata.stride[..dim - 3].iter())
            {
                let permute = if *stride == 0 {
                    0
                } else {
                    (i / stride) % shape
                };
                lhs_new_offset += permute * stride;
            }

            let lhs_new_shape = &lhs_metadata.shape[dim - 2..];
            let lhs_new_stride = &lhs_metadata.stride[dim - 2..];

            let mut rhs_new_offset = rhs_metadata.offset;
            for (shape, stride) in rhs_shape[..dim - 3]
                .iter()
                .zip(rhs_metadata.stride[..dim - 3].iter())
            {
                let permute = if *stride == 0 {
                    0
                } else {
                    (i / stride) % shape
                };
                rhs_new_offset += permute * stride;
            }

            let rhs_new_shape = &rhs_metadata.shape[dim - 2..];
            let rhs_new_stride = &rhs_metadata.stride[dim - 2..];

            let m = lhs_new_shape[0];
            let k = lhs_new_shape[1];
            let n = rhs_new_shape[1];

            let vec_offset = i * m * n;
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.,
                    lhs_metadata.data.as_ptr().add(lhs_new_offset),
                    lhs_new_stride[0] as isize,
                    lhs_new_stride[1] as isize,
                    rhs_metadata.data.as_ptr().add(rhs_new_offset),
                    rhs_new_stride[0] as isize,
                    rhs_new_stride[1] as isize,
                    0.,
                    vec.as_mut_ptr().add(vec_offset),
                    n as isize,
                    1,
                );
            }
        }

        let new_stride = shape_to_stride(&new_shape);

        let array = Array {
            data: vec,
            offset: 0,
            shape: new_shape,
            stride: new_stride,
        };

        Ok(array)
    }
}

pub trait OpsMatmulNDF64: OpsMatmul2DF64 {
    fn matmul_nd<Rhs>(&self, rhs: &Rhs) -> Result<Array<f64>, PzeudoNumErr>
    where
        Rhs: ArrayTrait<f64>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape.len() != rhs_metadata.shape.len() {
            return Err(MatmulNDErr(format!(
                "MatmulNDErr. OpsMatmulNDF32::matmul_nd. cannot perform matmul_nd on an array having shape {:?} and an array having shape {:?} due to unequal dimensions.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let lhs_shape = lhs_metadata.shape;
        let rhs_shape = rhs_metadata.shape;
        let dim = lhs_shape.len();

        if lhs_shape[..dim - 2] != rhs_shape[..dim - 2] {
            return Err(MatmulNDErr(format!(
                "MatmulNDErr. OpsMatmulNDF32::matmul_nd. cannot perform matmul_nd on an array having shape {:?} and an array having shape {:?} due to unequal batch sizes.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        if lhs_shape[dim - 1] != rhs_shape[dim - 2] {
            return Err(MatmulNDErr(format!(
                "MatmulNDErr. OpsMatmulNDF32::matmul_nd. cannot perform matmul_nd on an array having shape {:?} and an array having shape {:?} because it does not fulfill the shape '...×m×k * ...×k×n'.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let mut new_shape = lhs_shape[..dim - 2].to_vec();
        new_shape.push(lhs_shape[dim - 2]);
        new_shape.push(rhs_shape[dim - 1]);

        let mut vec = vec![0.; new_shape.iter().product::<usize>()];
        let matrix_count = lhs_shape[..dim - 2].iter().product::<usize>();
        for i in 0..matrix_count {
            let mut lhs_new_offset = lhs_metadata.offset;
            for (shape, stride) in lhs_shape[..dim - 3]
                .iter()
                .zip(lhs_metadata.stride[..dim - 3].iter())
            {
                let permute = if *stride == 0 {
                    0
                } else {
                    (i / stride) % shape
                };
                lhs_new_offset += permute * stride;
            }

            let lhs_new_shape = &lhs_metadata.shape[dim - 2..];
            let lhs_new_stride = &lhs_metadata.stride[dim - 2..];

            let mut rhs_new_offset = rhs_metadata.offset;
            for (shape, stride) in rhs_shape[..dim - 3]
                .iter()
                .zip(rhs_metadata.stride[..dim - 3].iter())
            {
                let permute = if *stride == 0 {
                    0
                } else {
                    (i / stride) % shape
                };
                rhs_new_offset += permute * stride;
            }

            let rhs_new_shape = &rhs_metadata.shape[dim - 2..];
            let rhs_new_stride = &rhs_metadata.stride[dim - 2..];

            let m = lhs_new_shape[0];
            let k = lhs_new_shape[1];
            let n = rhs_new_shape[1];

            let vec_offset = i * m * n;
            unsafe {
                matrixmultiply::dgemm(
                    m,
                    k,
                    n,
                    1.,
                    lhs_metadata.data.as_ptr().add(lhs_new_offset),
                    lhs_new_stride[0] as isize,
                    lhs_new_stride[1] as isize,
                    rhs_metadata.data.as_ptr().add(rhs_new_offset),
                    rhs_new_stride[0] as isize,
                    rhs_new_stride[1] as isize,
                    0.,
                    vec.as_mut_ptr().add(vec_offset),
                    n as isize,
                    1,
                );
            }
        }

        let new_stride = shape_to_stride(&new_shape);

        let array = Array {
            data: vec,
            offset: 0,
            shape: new_shape,
            stride: new_stride,
        };

        Ok(array)
    }
}
