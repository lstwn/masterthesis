# Incremental View Maintenance (IVM)

## Approaches

A high-level overview can be found [here][intro_to_ivm].

### PG_IVM

The traditional bag algebra approach but its susceptible to bottlenecks in a
data pipeline.

### Differential Dataflow (DD)

### DBSP

Successor to DD. It is less complex than DD and allows for out-of-order updates.
More can be found in [this HN thread][hn_dbsp_vs_dd].

**How does DBSP work multithreaded/distributed?** More info
[here][dist_dbsp_doc].

[intro_to_ivm]:
  https://materializedview.io/p/everything-to-know-incremental-view-maintenance
  "Everything You Need to Know About Incremental View Maintenance"
[hn_dbsp_vs_dd]: https://lobste.rs/s/5s4em5
[dist_dbsp_doc]: https://github.com/feldera/dist-design
