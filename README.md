# Serverless DAG Optimizer Core

Core implementation of risk-aware resource planning for serverless DAGs:
Lognormal fitting, measured correlation propagation, FW/Clark aggregation,
Repair/Prune, entry-cold risk estimation, JIT prewarming, and online Repair.

## Install

Requires Python 3.11 or later.

```bash
python -m pip install -e .
serverless-dag-core --help
```

Use `plan`, `risk`, or `entry-risk`; each command provides its own `--help`.
The Python APIs are under `src/serverless_dag/`.

## Inputs

Provide your own DAG, resource candidates, SLO, and profiling data. The model
directory uses these CSV files; their column names are defined in
`src/serverless_dag/planner/artifacts.py`:

- `per_stage_tier_lognormal_params.csv`
- `warm_tier_means.csv`
- `stage_latency_samples.csv`
- `stage_log_correlation.csv`
- `jit_warmup_lead_by_tier.csv`

Latency is in milliseconds, memory in MB, and execution cost in GB-s.
Benchmark actions, application DAGs, workloads, and experiment data are supplied
by the user; they are not bundled.

## OpenWhisk Integration

`integrations/openwhisk/core-integration.patch` adds container reservation,
pool-state queries, and Kubernetes readiness waiting. Apply it to OpenWhisk
revision `ef725a653ab112391f79c274d8e3dcfb915d59a3`.
Actions must report `action_duration_ms`, handle `__warmup`, and preserve
`__ow_reservation_key` for JIT reservation reuse. Cluster deployment and replay
orchestration are supplied separately.

## License

The framework source in `src/` and its documentation are licensed under the
[MIT License](LICENSE). The OpenWhisk integration patch retains Apache-2.0
licensing and upstream notices in `integrations/openwhisk/`.
