use dbsp::{NestedCircuit, RootCircuit};

enum Circuit {
    Root(RootCircuit),
    Nested(NestedCircuit),
}
