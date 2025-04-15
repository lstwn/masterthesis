# DBSP Notes

## Operations on Streams

| Function          | Linear        | RootCircuit | NestedCircuit | Arbitrary        |
| ----------------- | ------------- | ----------- | ------------- | ---------------- |
| `sum`/`plus`      | yes           | yes         | yes           | yes              |
| `delta0`          | ?             | yes         | yes           | yes              |
| `filter`          | yes           | yes         | yes           | no (only `dyn_`) |
| `map_index`/`map` | yes           | yes         | yes           | no (only `dyn_`) |
| `join`            | no (bilinear) | yes         | yes           | no (only `dyn_`) |
| `distinct`        | no            | yes         | yes           | yes?             |
| `aggregate`       | no            | yes         | yes           | yes?             |
| `output`          | -             | yes         | no            | no               |

## Operations on Circuits

| Function        | RootCircuit | NestedCircuit | Arbitrary |
| --------------- | ----------- | ------------- | --------- |
| `recursive`     | yes         | yes           | yes       |
| `add_input_...` | yes         | no            | no        |
