use dbsp::{NestedCircuit, RootCircuit};

#[deprecated]
enum Circuit {
    Root(RootCircuit),
    Nested(NestedCircuit),
}
