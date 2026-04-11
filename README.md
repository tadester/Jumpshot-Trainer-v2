# JumpShot Trainer

JumpShot Trainer is a Rust-first biomechanical jump-shot analyzer with a Python video-ingestion pipeline.

It takes raw shooting footage, extracts pose and ball signals, segments shots, computes mechanics, exports a shared Parquet corpus, and opens everything in a native Rust desktop review app.

The desktop experience is built around one main flow:

1. launch the app
2. drop in a video or paste a video path
3. choose the camera angle
4. click `Analyze Video`
5. get coaching feedback on what is wrong and what to adjust

Suggested GitHub repository description:

`Rust biomechanics shooting trainer with a Python video-ingestion pipeline, MediaPipe + YOLOv8 teacher models, Parquet feature exports, and a desktop analysis dashboard.`

## What This Project Is

The project has two main parts:

- `janitor_python/`
  Ingests raw video, stores manifests, runs teacher models, segments shots, extracts biomechanics features, and exports Parquet.
- `athlete_rust/`
  Loads the processed corpus, fits a lightweight supervised score layer, and provides the desktop calibration and review experience.

In practice, the flow is:

1. Record shooting footage from side and front-quarter angles.
2. Intake and process the videos with the Python janitor.
3. Rebuild the shared training corpus.
4. Launch the Rust desktop app and analyze a clip from a focused upload-and-review screen.

## Current Capabilities

Working today:

- raw upload intake
- per-clip manifest generation
- MediaPipe + YOLOv8 teacher pipeline with fallbacks
- automatic shot segmentation
- automatic rescue on previously weak sessions
- biomechanics feature extraction
- Parquet corpus generation
- side/front pairing heuristics
- focused desktop upload-and-analyze flow
- per-shot coaching review
- lightweight supervised Rust-side scoring

Current extracted mechanics include:

- elbow flexion
- knee load
- forearm verticality
- elbow flare
- release height ratio
- release timing
- release vs apex offset
- jump height

## Quick Start

### 1. Set Up Python

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/janitor_python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Make Sure Model Assets Exist

Recommended local assets:

- `yolov8n.pt`
- `yolov8n-pose.pt`
- `datasets/models/mediapipe/pose_landmarker_lite.task`

### 3. Put Videos In The Inbox

- side clips:
  `/Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/datasets/uploads/inbox/side`
- front-quarter clips:
  `/Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/datasets/uploads/inbox/front`

### 4. Intake A Clip

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor intake-video \
  --project-root . \
  --clip /absolute/path/to/clip.mp4 \
  --view side
```

Use `--view angle45` for the front-quarter camera.

### 5. Process A Session

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor strong-process \
  --project-root . \
  --manifest datasets/uploads/manifests/<manifest>.json \
  --athlete-profile datasets/calibration_20_shot/annotations/athlete_profile.json \
  --source-dataset uploaded_session \
  --teacher-model mediapipe_yolov8_teacher \
  --frame-stride 30 \
  --yolo-weights yolov8n.pt \
  --pose-weights yolov8n-pose.pt \
  --mediapipe-model datasets/models/mediapipe/pose_landmarker_lite.task
```

Optional last-resort rescue path:

```bash
janitor_python/.venv/bin/jumpshot-janitor strong-process \
  --project-root . \
  --manifest datasets/uploads/manifests/<manifest>.json \
  --athlete-profile datasets/calibration_20_shot/annotations/athlete_profile.json \
  --source-dataset uploaded_session \
  --teacher-model mediapipe_yolov8_teacher \
  --frame-stride 30 \
  --yolo-weights yolov8n.pt \
  --pose-weights yolov8n-pose.pt \
  --mediapipe-model datasets/models/mediapipe/pose_landmarker_lite.task \
  --tuning datasets/uploads/tuning/<session>.json
```

### 6. Rebuild The Shared Corpus

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor build-corpus --project-root .
```

This writes:

- `datasets/shared/processed/training_corpus.parquet`
- `datasets/shared/processed/training_corpus.metadata.json`

### 7. Run The Rust App

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/athlete_rust
cargo run
```

When the app opens:

1. drag a video into the window or paste its full path
2. choose `Side View` or `Front Quarter`
3. click `Analyze Video`
4. review the detected shots, coaching adjustments, phase feedback, and mechanical snapshot

### 8. Verify Everything Builds

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/python -m py_compile \
  janitor_python/src/jumpshot_janitor/cli.py \
  janitor_python/src/jumpshot_janitor/video_pipeline.py \
  janitor_python/src/jumpshot_janitor/exporters.py

cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/athlete_rust
cargo check
```

## How The App Works

### Video Intake

The janitor stores:

- raw clip copy
- manifest JSON
- fps
- resolution
- duration
- orientation

This gives the repo a stable raw-video archive under `datasets/uploads/`.

### Teacher Extraction

The current teacher stack is:

- YOLOv8 for basketball detection
- MediaPipe pose when it initializes
- YOLO pose fallback
- built-in pose heuristic fallback if stronger pose extraction is unavailable

### Shot Segmentation

The current segmentation stack is layered:

- primary ball-hand separation logic
- multi-signal fallback using wrist lift, release height, and jump motion
- wrist-only fallback
- cycle and valley fallbacks
- manual seeding only when automatic segmentation still fails

This matters because weak sessions are no longer forced into manual tagging first.

### Feature Extraction

For each shot, Python computes structured biomechanics features and exports them into Parquet.

Rust then reads those records and uses them for:

- diagnostics
- session summaries
- processed-shot browsing
- model-readiness status
- lightweight supervised scoring

## What “Trained” Means Right Now

This is not yet a fully trained end-to-end production vision system.

Right now the project is:

- teacher-driven for perception
- feature-driven for biomechanics
- supervised only at the Rust structured-feature scoring layer

So the current model stack is:

1. Teacher models detect pose and ball signals.
2. Geometry and temporal logic convert those into biomechanics features.
3. Rust reads the resulting feature corpus and fits a lightweight supervised score model.

That means the app is already useful, but it is not yet the final long-term learned biomechanics engine.

## Training Status

Today, the system is best described as:

- teacher-labeled
- feature-driven
- lightly supervised on the structured corpus

Long-term improvements still wanted:

- larger multi-athlete corpus
- more paired side/front sessions
- more trusted gold annotations
- stronger stage ground truth
- a true Rust-side Candle training pass on normalized tensors

## What The Rust App Shows

The Rust desktop app currently includes:

- a focused upload panel
- background processing status
- per-shot review after analysis
- coaching adjustments
- phase feedback
- mechanical snapshot cards
- a visual overlay review panel
- hidden engine details for corpus and model background status

## Current Repo State

Validated recently:

- 4 uploaded videos were ingested and processed
- the two previously weak sessions now auto-segment without manual seeds
- the shared corpus contains uploaded-session rows and paired-view rows
- only a small manual-stage-tagged remainder is still in the corpus
- Python compile checks pass
- Rust `cargo check` passes

Still being tuned:

- shot count realism on long sessions
- release timing realism on sparse-stride runs
- pairing quality between independently recorded sessions
- broader generalization to more athletes and camera setups

## Folder Overview

- `athlete_rust/`
  Rust desktop application and analysis engine.
- `janitor_python/`
  Python ingestion and export pipeline.
- `datasets/uploads/`
  Raw session archive, manifests, processed sessions, inbox, and tuning files.
- `datasets/calibration_20_shot/`
  Manual or semi-manual gold validation data.
- `datasets/shared/processed/`
  Shared Parquet corpus read by Rust.
- `datasets/models/mediapipe/`
  Local MediaPipe task models.
- `schemas/`
  Shared schema contracts.
- `docs/`
  Architecture and product notes.

## Troubleshooting

- If MediaPipe fails on macOS, the janitor should fall back automatically instead of crashing.
- If a clip still does not segment cleanly, use the `--tuning` path with a session JSON under `datasets/uploads/tuning/`.
- If the Rust app opens with older data, rebuild the corpus first.

## Project Goal

The goal is a real Rust desktop jump-shot trainer that can eventually analyze arbitrary athletes from uploaded video, compare mechanics over time, and grow into a stronger supervised biomechanics system without discarding the current practical MVP pipeline.
