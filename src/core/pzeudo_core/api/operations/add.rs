use crate::{pzeudo_core::core::PzeudoCore, tensor_core::Tensor};

impl<const N: usize, const MAX_RING_BUFFER: usize> PzeudoCore<N, MAX_RING_BUFFER> {
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let execute_ops = self.tensor_core.add_execute(a, b);
        let tensor =
            self.thread_pool
                .executor(execute_ops, a.inner.borrow_mut(), b.inner.borrow_mut());

        tensor
    }
}
