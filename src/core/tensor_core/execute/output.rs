use cahotic::OutputTrait;

pub struct ExecuteOut(pub Result<f64, ()>);

impl OutputTrait for ExecuteOut {}
