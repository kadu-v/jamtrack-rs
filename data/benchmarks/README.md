# Benchmark data

Benchmark results are grouped by tracker family. Each JSON file is standalone
and uses `schema_version: 1`. A file may contain two benchmark sections:

- `performance`: elapsed time for the local fixed-frame benchmark, including
  device, input, and frame-count context.
- `mot17_train`: TrackEval metrics for MOT17-train, including the detector used.

Every result records a display `label`, `implementation`, and `variant` so new
Rust/Python or tuned/ECC variants can be added without changing the chart
generator. MOT17 results also provide a compact `chart_label` for annotations.
Python reference results declare `official: true`.

Regenerate all charts from these files with:

```bash
uv run --project python python scripts/gen_charts.py
```
