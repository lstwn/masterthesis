# Implementation

## Quick Start

1. Install and set up your workspace in [devpod](https://devpod.sh).
2. Run `devpod up` to start the development container and open the workspace
   in your configured editor.
3. Then:
   - Run `cargo build` to compile.
   - Run `cargo test` to run tests.
   - Run `cargo run` to start the application.

## Work Outline

For proof-of-concept:

- [x] selection
- [x] projection
- [x] [nice cli table output](https://crates.io/crates/cli-table)
      for results through introduction of input/output schema?
- [ ] join
  - [x] plain case
  - [x] add support for projection within join
  - [ ] with same table (requires aliasing)
  - [ ] try out adding missing "link edge" incrementally
- [ ] test multithreaded environment
- [ ] iteration until fix point
- [ ] add Datalog parser for string representation

Benefits of a code gen pass taking ownership of the AST:

- A pointer based AST can be transformed into a flattened AST before execution
  https://www.cs.cornell.edu/~asampson/blog/flattening.html

Benefits of a type checker pass which immutably references the AST:

- It can check the types of expressions and statements
