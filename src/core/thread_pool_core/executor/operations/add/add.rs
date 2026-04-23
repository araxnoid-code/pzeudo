use cahotic::{DefaultOutput, ScheduleVec};

use crate::ExecuteOut;

pub fn add_executor(
    a: &Option<f64>,
    b: &Option<f64>,
    schedule_vec: Option<ScheduleVec<ExecuteOut>>,
) -> ExecuteOut {
    println!("execution running on");
    let (a, b) = if let Some(schedule) = schedule_vec {
        let mut index_access = 0;

        let a = if let Some(value) = a {
            *value
        } else if let Some(sch_out) = schedule.get(index_access) {
            if let Ok(value) = sch_out.0 {
                index_access += 1;
                value
            } else {
                panic!()
            }
        } else {
            panic!()
        };

        let b = if let Some(value) = b {
            *value
        } else if let Some(sch_out) = schedule.get(index_access) {
            if let Ok(value) = sch_out.0 {
                value
            } else {
                panic!()
            }
        } else {
            panic!()
        };

        (a, b)
    } else {
        let a = a.unwrap();
        let b = b.unwrap();

        (a, b)
    };

    println!("execution done {} + {} = {}", a, b, a + b);
    ExecuteOut(Ok(a + b))
}
