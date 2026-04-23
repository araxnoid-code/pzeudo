use std::{cell::RefCell, sync::Arc};

use cahotic::{
    Cahotic, CahoticBuilder, DefaultOutput, DefaultSchedule, DefaultTask, Schedule, ScheduleVec,
    SchedulerTrait, TaskTrait,
};

fn main() {
    let cahotic = CahoticBuilder::default::<f64>()
        .set_task_type::<ExecuteOps>()
        .set_schedule_type::<ExecuteOps>()
        .build()
        .unwrap();

    let mut sch_list = vec![];

    let tensor_a = Tensor::init(10.);
    let tensor_b = Tensor::init(20.);

    let tensor_c = tensor_a.add(&tensor_b, &cahotic, &mut sch_list);

    let tensor_d = Tensor::init(30.);

    let tensor_e = tensor_d.add(&tensor_c, &cahotic, &mut sch_list);

    executor(&cahotic, sch_list);

    cahotic.join();
}

fn executor(
    cahotic: &Cahotic<ExecuteOps, ExecuteOps, DefaultOutput<f64>, 4, 4096>,
    sch_list: Vec<Arc<RefCell<TensorType>>>,
) {
    for sch in (sch_list).into_iter().rev() {
        if let TensorType::Schedule(sch) = &mut *sch.borrow_mut() {
            let sch = sch.take().unwrap();

            cahotic.schedule_exec(sch);
        }
    }
}

enum TensorType {
    Initial(f64),
    Schedule(Option<Schedule<ExecuteOps, ExecuteOps, DefaultOutput<f64>>>),
}

impl TensorType {
    pub fn get_initial(&self) -> Option<f64> {
        match self {
            Self::Initial(value) => Some(*value),
            Self::Schedule(_) => None,
        }
    }
}

struct Tensor {
    inner: Arc<RefCell<TensorType>>,
}

impl Tensor {
    pub fn init(value: f64) -> Tensor {
        Tensor {
            inner: Arc::new(RefCell::new(TensorType::Initial(value))),
        }
    }

    pub fn sch(schedule: Schedule<ExecuteOps, ExecuteOps, DefaultOutput<f64>>) -> Tensor {
        Tensor {
            inner: Arc::new(RefCell::new(TensorType::Schedule(Some(schedule)))),
        }
    }

    pub fn add(
        &self,
        other: &Tensor,
        cahotic: &Cahotic<ExecuteOps, ExecuteOps, DefaultOutput<f64>, 4, 4096>,
        sch_list: &mut Vec<Arc<RefCell<TensorType>>>,
    ) -> Tensor {
        let execute_ops = ExecuteOps::AddExecute(
            self.inner.borrow().get_initial(),
            other.inner.borrow().get_initial(),
        );

        if let (TensorType::Initial(_), TensorType::Initial(_)) =
            (&*self.inner.borrow(), &*other.inner.borrow())
        {
            let schedule = cahotic.scheduling_create_initial(execute_ops);
            let output = Tensor::sch(schedule);
            sch_list.push(output.inner.clone());
            output
        } else {
            let mut schedule = cahotic.scheduling_create_schedule(execute_ops);

            if let TensorType::Schedule(dep) = &mut *self.inner.borrow_mut() {
                cahotic
                    .schedule_after(&mut schedule, dep.as_mut().unwrap())
                    .unwrap();
            }

            if let TensorType::Schedule(dep) = &mut *other.inner.borrow_mut() {
                cahotic
                    .schedule_after(&mut schedule, dep.as_mut().unwrap())
                    .unwrap();
            }

            let output = Tensor::sch(schedule);
            sch_list.push(output.inner.clone());
            output
        }
    }

    pub fn sub(
        &self,
        other: &Tensor,
        cahotic: &CahoticBuilder<
            ExecuteOps,
            DefaultSchedule<DefaultOutput<f64>>,
            DefaultOutput<f64>,
            4,
            4096,
        >,
    ) {
    }
}

enum ExecuteOps {
    AddExecute(Option<f64>, Option<f64>),
    SubExecute(Option<f64>, Option<f64>),
}

impl SchedulerTrait<DefaultOutput<f64>> for ExecuteOps {
    fn execute(&self, scheduler_vec: ScheduleVec<DefaultOutput<f64>>) -> DefaultOutput<f64> {
        match self {
            Self::AddExecute(a, b) => add_executor(a, b, Some(scheduler_vec)),
            ExecuteOps::SubExecute(a, b) => DefaultOutput(10.),
        }
    }
}

impl TaskTrait<DefaultOutput<f64>> for ExecuteOps {
    fn execute(&self) -> DefaultOutput<f64> {
        match self {
            Self::AddExecute(a, b) => add_executor(a, b, None),
            ExecuteOps::SubExecute(a, b) => DefaultOutput(10.),
        }
    }
}

fn add_executor(
    a: &Option<f64>,
    b: &Option<f64>,
    schedule_vec: Option<ScheduleVec<DefaultOutput<f64>>>,
) -> DefaultOutput<f64> {
    if let Some(sch) = schedule_vec {
        let mut index_access = 0;

        let a = if let Some(value) = a {
            *value
        } else if let Some(DefaultOutput(value)) = sch.get(index_access) {
            index_access += 1;
            *value
        } else {
            panic!("")
        };

        let b = if let Some(value) = b {
            *value
        } else if let Some(DefaultOutput(value)) = sch.get(index_access) {
            *value
        } else {
            panic!("")
        };

        println!("done execute {}", a + b);
        DefaultOutput(a + b)
    } else {
        let a = a.unwrap();
        let b = b.unwrap();

        println!("done execute {}", a + b);

        DefaultOutput(a + b)
    }
}
