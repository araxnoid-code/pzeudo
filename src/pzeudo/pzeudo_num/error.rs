#[derive(Debug)]
pub enum PzeudoNumErr {
    // tools
    ShapeToStride(&'static str),

    // Array
    ArrayNewErr(&'static str),
}
