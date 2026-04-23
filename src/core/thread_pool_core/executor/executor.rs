use std::cell::RefMut;

use crate::{
    ThreadPoolCore,
    tensor_core::{ExecuteOps, InnerTensor, Tensor},
};

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    /// executor, functions to execute ExecuteOps that have been created by TensorCore
    pub(crate) fn executor(
        &mut self,
        execute_ops: ExecuteOps,
        mut inner_tensor_a: RefMut<'_, InnerTensor>,
        mut inner_tensor_b: RefMut<'_, InnerTensor>,
    ) -> Tensor {
        if let (InnerTensor::Initial(_), InnerTensor::Initial(_)) =
            (&*inner_tensor_a, &*inner_tensor_b)
        {
            let schedule = self.executor.scheduling_create_initial(execute_ops);
            let output = Tensor::sch(schedule);
            self.schedule_order.push(output.inner.clone());
            output
        } else {
            let mut schedule = self.executor.scheduling_create_schedule(execute_ops);

            if let InnerTensor::Schedule(dep) = &mut *inner_tensor_a {
                self.executor
                    .schedule_after(&mut schedule, dep.as_mut().unwrap())
                    .unwrap();
            }

            if let InnerTensor::Schedule(dep) = &mut *inner_tensor_b {
                self.executor
                    .schedule_after(&mut schedule, dep.as_mut().unwrap())
                    .unwrap();
            }

            let output = Tensor::sch(schedule);
            self.schedule_order.push(output.inner.clone());
            output
        }
    }

    /// execute_schedule, all schedule orders that have been created via the ThreadPoolCore::executor method
    pub(crate) fn execute_schedule(&mut self) {
        for sch in self.schedule_order.iter().rev() {
            if let InnerTensor::Schedule(sch) = &mut *sch.borrow_mut() {
                let schedule = sch.take().unwrap();
                self.executor.schedule_exec(schedule);
            }
        }

        self.schedule_order.clear();
    }
}
