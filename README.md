# JumpShot Trainer

JumpShot Trainer is a two-part shooting-form analysis system:

- `janitor_python/`
  The video and data-engineering side. It ingests raw clips, runs teacher-model extraction, segments shots, computes frame and shot features, and exports Parquet.
- `athlete_rust/`
  The biomechanics and product side. It loads processed corpora, calibrates the athlete, scores mechanics, evaluates training readiness, and presents the review experience in a desktop app.

The intended workflow is:

1. upload raw clips into the Python janitor
2. run a teacher backend such as MediaPipe + YOLOv8
3. export processed frame/shot tables and a shared training corpus
4. load the corpus in the Rust athlete app for review, scoring, and future training

## Repository Description

Suggested GitHub repository description:

`Rust biomechanics shooting trainer with a Python video-ingestion pipeline, MediaPipe + YOLOv8 teacher models, Parquet feature exports, and a desktop analysis dashboard.`

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

## Current Capabilities

- raw video intake with metadata extraction
- built-in upload/session manifest storage
- teacher-model processing paths:
  - OpenCV fallback teacher
  - MediaPipe pose + YOLOv8 ball detection teacher
- automatic shot segmentation from processed frame observations
- shot-level biomechanical feature extraction
- Parquet export for calibration sets and shared training corpora
- Rust desktop review UI with calibration, diagnostics, visual review, and session audit
- training-readiness scoring for the current corpus

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

Build a broader training corpus across every dataset folder that contains `annotations/athlete_profile.json` and `annotations/shots.csv`:

```bash
cd janitor_python
source .venv/bin/activate
jumpshot-janitor build-corpus --project-root ..
```

That writes:

- `datasets/shared/processed/training_corpus.parquet`
- `datasets/shared/processed/training_corpus.metadata.json`

Upload and process a raw clip:

```bash
cd janitor_python
source .venv/bin/activate
jumpshot-janitor intake-video --project-root .. --clip /absolute/path/to/clip.mp4 --view side
```

That stores the raw video and writes a manifest under `datasets/uploads/manifests/`.

Then process it with teacher-model outputs:

```bash
jumpshot-janitor process-session \
  --project-root .. \
  --manifest /absolute/path/to/manifest.json \
  --athlete-profile ../datasets/calibration_20_shot/annotations/athlete_profile.json \
  --pose-json /absolute/path/to/pose_frames.json \
  --ball-json /absolute/path/to/ball_tracks.json \
  --source-dataset uploaded_session \
  --teacher-model teacher_import
```

That writes frame observations and shot records into `datasets/uploads/processed/<session_id>/`, and those processed session shot records are automatically folded into `build-corpus`.

For a built-in one-command path using the OpenCV teacher backend:

```bash
jumpshot-janitor auto-process \
  --project-root .. \
  --manifest /absolute/path/to/manifest.json \
  --athlete-profile ../datasets/calibration_20_shot/annotations/athlete_profile.json \
  --source-dataset uploaded_session \
  --teacher-model builtin_cv_teacher \
  --frame-stride 2
```

That will:

1. detect the athlete with OpenCV's built-in people detector
2. detect the basketball with an orange-ball heuristic
3. generate pseudo-pose landmarks frame by frame
4. segment shot windows automatically
5. write frame observations, teacher JSON, session JSON, and shot-record Parquet

For the stronger teacher path using MediaPipe pose + YOLOv8 basketball detection:

```bash
jumpshot-janitor strong-process \
  --project-root .. \
  --manifest /absolute/path/to/manifest.json \
  --athlete-profile ../datasets/calibration_20_shot/annotations/athlete_profile.json \
  --source-dataset uploaded_session \
  --teacher-model mediapipe_yolov8_teacher \
  --frame-stride 2 \
  --yolo-weights yolov8n.pt
```

The first run may download YOLOv8 weights automatically.

Typical upload flow:

```bash
cd janitor_python
source .venv/bin/activate
jumpshot-janitor intake-video --project-root .. --clip /absolute/path/to/clip.mp4 --view side
jumpshot-janitor strong-process --project-root .. --manifest /absolute/path/to/manifest.json --athlete-profile ../datasets/calibration_20_shot/annotations/athlete_profile.json --source-dataset uploaded_session --teacher-model mediapipe_yolov8_teacher --frame-stride 2 --yolo-weights yolov8n.pt
jumpshot-janitor build-corpus --project-root ..
```

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

When the shared Parquet exists, the Rust app will automatically prefer `training_corpus.parquet` on startup and fall back to the calibration set if no wider corpus has been exported yet.

## Validation

- `cargo check` in `athlete_rust/`
- Parquet export and corpus export from `janitor_python/`
- Python teacher/session processing modules compile cleanly

## Notes

- The built-in OpenCV teacher is a fallback path.
- The stronger production-oriented teacher path is `MediaPipe + YOLOv8`.
- The 20-shot calibration set should be treated as gold-tier validation data, not the full long-term training corpus.

## Recent Changes

- The Rust desktop UI now includes a calibration deck, performance dashboard, visual review panels, and a training-readiness surface.
- The Rust core now builds normalized training examples from janitor Parquet and scores whether the current dataset is ready for the first supervised training run.
- The Python janitor now supports raw-video intake, session manifests, processed upload sessions, and a stronger MediaPipe + YOLOv8 teacher backend.
