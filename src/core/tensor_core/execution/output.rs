use cahotic::OutputTrait;

pub struct ExecutionOutput(pub Result<i32, ()>);
impl OutputTrait for ExecutionOutput {}
