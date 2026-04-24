use std::cell::RefMut;

use crate::{
    TensorValue, ThreadPoolCore,
    tensor_core::{ExecuteOps, InnerTensorType, Tensor},
};

impl<const N: usize, const MAX_RING_BUFFER: usize> ThreadPoolCore<N, MAX_RING_BUFFER> {
    /// executor, functions to execute ExecuteOps that have been created by TensorCore
    pub(crate) fn executor(
        &mut self,
        execute_ops: ExecuteOps,
        mut inner_tensor_a: RefMut<'_, InnerTensorType>,
        mut inner_tensor_b: RefMut<'_, InnerTensorType>,
    ) -> Tensor {
        if let (InnerTensorType::Initial(_), InnerTensorType::Initial(_)) =
            (&*inner_tensor_a, &*inner_tensor_b)
        {
            let schedule = self.executor.scheduling_create_initial(execute_ops);
            let output = Tensor::sch(schedule);
            self.schedule_order.push(output.inner.clone());
            output
        } else {
            let mut schedule = self.executor.scheduling_create_schedule(execute_ops);

            if let InnerTensorType::Schedule(dep) = &mut *inner_tensor_a {
                self.executor
                    .schedule_after(&mut schedule, dep.schedule.as_mut().unwrap())
                    .unwrap();
            }

            if let InnerTensorType::Schedule(dep) = &mut *inner_tensor_b {
                self.executor
                    .schedule_after(&mut schedule, dep.schedule.as_mut().unwrap())
                    .unwrap();
            }

            let output = Tensor::sch(schedule);
            self.schedule_order.push(output.inner.clone());
            output
        }
    }

    /// execute_schedule, all schedule orders that have been created via the ThreadPoolCore::executor method
    pub(crate) fn execute_schedule(&mut self) {
        for inner_tensor_type in self.schedule_order.iter().rev() {
            if let InnerTensorType::Schedule(inner_tensor) = &mut *inner_tensor_type.borrow_mut() {
                let schedule = inner_tensor.schedule.take().unwrap();
                let poll = self.executor.schedule_exec(schedule);
                inner_tensor.data = TensorValue::Poll(poll);
            }
        }

        self.schedule_order.clear();
    }
}
