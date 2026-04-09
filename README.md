# JumpShot Trainer

This repo is now split into two coordinated workspaces:

- `janitor_python/`
  The data-engineering side. It ingests annotations, prepares shot records, and exports Parquet.
- `athlete_rust/`
  The biomechanics, ML scoring, desktop UI, and Rust-side ingestion app.

## Folder Structure

- `athlete_rust/`
  Rust desktop app and analysis engine.
- `janitor_python/`
  Python ingestion/export pipeline.
- `datasets/`
  Raw clips, annotations, derived files, and shared Parquet.
- `schemas/`
  Shared table contracts between Python and Rust.
- `docs/`
  Product and system notes.

## Janitor Pipeline

Build Parquet from your annotation templates:

```bash
cd janitor_python
python -m venv .venv
source .venv/bin/activate
pip install -e .
jumpshot-janitor build-parquet --project-root .. --dataset calibration_20_shot
```

That writes:

- `datasets/calibration_20_shot/derived/shot_records.parquet`
- `datasets/shared/processed/calibration_20_shot_shot_records.parquet`

## Athlete App

Run the Rust desktop trainer:

```bash
cd athlete_rust
cargo run
```

Verify the Rust workspace:

```bash
cd athlete_rust
cargo check
```

When the shared Parquet exists, the Rust app will automatically load it on startup and seed the trainer from the first shot record.

## What Changed

- The Rust desktop UI now includes a calibration deck, performance dashboard, visual review panels, and a training-readiness surface.
- The Rust core now builds normalized training examples from janitor Parquet and scores whether the current dataset is ready for the first supervised training run.
