# Version 0.0.4
## Memperbaiki Bug pada ArrayTrait::index
Bug ditemukan pada method ArrayTrait::index. Bug terjadi karena kesalahan pada kalkulasi stride yang menyababkan kesalahan saat digunakan pada ArrayTrait::index pada ArrayView atau Array bersifat View lainnya. Bug ini telah diperbaiki.

ArrayTrait::index telah ditambahkan validasi tambahan untuk memastikan indexing masih di dalam range dimensi tensor.

## Memperbaiki Bug pada code testing pzeudo_num::_test::matmul::matmul_2d::matmul_2d_test_3
Bug terjadi karena kesalahan indexing dikarenakan ArrayTrait::index tidak memvalidasi indexing yang melewati batas array.

## Penambahan BatchNorm
```rs
fn main() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let batch_norm = BatchNorm::new(2, 16, &mut model_builder).unwrap();
}
```
Untuk Setiap paramaters pada BatchNorm::new
```rs
pub fn new(
        channel: usize, // index dimension that becomes a channel on the input
        channel_size: usize, // size of channel to be input
        model_builder: &mut ModelBuilder<F>,
    ) -> Result<BatchNorm<F>, PzeudoErr>
    where
        F: NumCast,
    {
        // ...
    }
```
#### Default Value
```
gamma = ones tensor
beta = zeros tensor
momentum = 0.1
running_avg = 0
running_var = 1
```

#### Formula
```
avg = E[x]
variance = E[x^2] - E[x]^2
epsilon = 1e-7
norm = x - avg/sqrt(variance + epsilon)
y = norm * gamma + beta
```

#### Training Phase
Saat training phase. BatchNorm akan melakukan kalkulasi mencari mean dan variance melalui tensor input lalu menggunakannya untuk menormalisasikan tensor dan update running_avg serta running_var.
```
momentum = 0.1 (default)
running_avg = (1 - momentum) * running_avg + momentum * avg
running_var = (1 - momentum) * running_var + momentum * var
```

#### Testing Phase
Saat testing phase. BatchNorm akan menggunakan running_avg dan running_var untuk menormalisasikan tensor.
```
norm = x - running_avg/sqrt(running_var + epsilon)
y = norm * gamma + beta
```

## Menambahkan trait OpsMax dan OpsMin residual baru
- OpsMax::max
- OpsMax::max_axis
- OpsMax::argmax
- OpsMax::argmax_axis
- OpsMax::max_axis_with_flatten_index
- OpsMin::min
- OpsMin::min_axis
- OpsMin::argmin
- OpsMin::argmin_axis
- OpsMin::min_axis_with_flatten_index

## Menambahkan metode residual baru pada tensor
#### Tensor::max
Mengembalikan value terbesar dari sebuah tensor.

#### Tensor::max_axis
Mengembalikan value terbesar dari sebuah tensor berdasarkan axisnya.

#### Tensor::argmax
Mengembalikan index dari value terbesar dari sebuah tensor.
Method ini tidak dapat melakukan backprpogation.

#### Tensor::argmax_axis
Mengembalikan index dari value terbesar dari sebuah tensor berdasarkan axisnya.
Method ini tidak dapat melakukan backprpogation.

## Method Inisialisasi Array Baru
- Array::from_range
- Array::from_range_fn
- Array::from_shape
- Array::from_shape_fn

## Mengimplementasikan SlicingRangeTrait untuk usize


## Perubahan pada LayerNorm
Kini LayerNorm akan secara langsung berstatus ReqGrad tanpa harus notasi manual saat inisialisasinya.

## Menambahkan Method alg_optim untuk setiap optimizer
alg_optim adalah method yang menggunakan algebraic dalam operasinya. pengembangannya di tangguhkan untuk kedepannya dan tidak menjadi prioritas utama.

## Menambahkan Tensor::dot
