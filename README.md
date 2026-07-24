# Pzeudo
a deep learning project for fun.

## Goal
can be used to create AI models, that's all.

## Stable?
still far from stable, but will continue to be developed.

## what's new in 0.0.1-dev.6
Update 0.0.1-dev.6 berfokus pada pengembangan alur backpropogation dan pembuatan model deep learning. Perubahan yang terjadi antara lain:

#### fix bugs
- mengatasi bug pada Array::matmul_2d dikarenakan kesalahan pada offset.
- Menyelesaikan masalah pada function get_broadcast_dim yang tidak memberikan dimensi yang di broadcast secara berurut.

#### New
- Penambahan Auto broadcast pada perhitungan 
  - Add
  - Sub
  - Mul
  - Div

- penambahan metode matmul_2d(untuk f32 dan f64) pada tensor

- Penambahan struct Module, yang akan menjadi environment utama yang menyediakan source untuk keperluan deep learning.

- Menambahkan EpochBuilder, yang akan menjadi struktur utama dalam configurasi proses training nantinya.

- Mekanisme training akan melalui Module::epoch.

- penambahan metode untuk Array
  - Array::to_ones
  - Array::to_zeros

- Penambahan Optimazer
  - Sgd

- Penambahan Loss Function
  - Mean Squere Error(MSE).

- Menambahkan method linear yang memungkinkan untuk membuat linear layer.

- perubahan cara menyimpan array tensor, kini ada 2 cara menyimpan array pada ArrayStorage, menggunakan storage biasa dan permanent_storage
  - storage, diperlukan untuk memungkinkan menyimpan array sementara yang diperlukan untuk alur backpropogation. untuk memanage memory maka storage akan dihapus secara berkala
    - pengahpusan storage secara otomatis per epoch melalui Module::epoch, setelah penghapusan maka storage siap digunakan untuk epoch berikutnya tanpa menyimpan data yang tidak perlu dari epoch sebelumnya.
  - permanent_storage, permanent_storage tidak diperuntukan untuk dihapus karena berfungsi meyimpan array secara permanent
    - penyimpanan array secara permanent diperlukan untuk menyimpan array yang akan diupdate oleh optimazer.

- Perubahan pada ArrayStorage yang kini memungkinkan menyimpan array updateable(untuk keperluan optimazer), array view dan array serta mengurangi hubungan yang tidak perlu diantara array array yang disimpan.
  - Dikarenakan perubahan ini, struktur yang menyimpan index array mengalami perubahan dalam cara menyimpan index dan handle output dari storage, namun secara alur kebanyakan dari struktur tersebut tidak berubah.

lihat proses pengembangan lebih detail pada: [0.0.1-dev.6_plan.md](0.0.1-dev.6_plan.md)
