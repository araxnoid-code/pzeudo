/// GraphType
/// When tensors are executed, a graph is formed that allows backpropagation. `GraphType` itself is used to create a schedule on the chaotic based on the graph that is formed.
pub(crate) enum GraphType {
    Initial,
    Schedule,
}
