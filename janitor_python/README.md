# Janitor Python

This workspace handles the data-engineering side of the project.

Responsibilities:

- ingest athlete profile and shot annotations
- prepare a clean training/analysis table
- export Parquet for the Rust athlete app
- later host teacher-model and auto-label integrations

## Quick Start

```bash
cd janitor_python
python -m venv .venv
source .venv/bin/activate
pip install -e .
jumpshot-janitor build-parquet \
  --project-root .. \
  --dataset calibration_20_shot
```
