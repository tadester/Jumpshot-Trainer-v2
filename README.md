# JumpShot Trainer

JumpShot Trainer is a desktop jump-shot review app backed by a Python video-processing pipeline.

You give it a shooting clip, choose the camera angle, and the system:

- ingests the raw video into the project workspace
- runs pose and ball tracking with the janitor pipeline
- segments detected shots
- extracts structured biomechanics features
- rebuilds a shared Parquet training corpus
- opens the result in a native Rust review UI

The current app is built to be useful right now as a visual coaching tool, not just a research prototype.

Suggested repository description:

`Desktop jump-shot analysis app with a Rust review UI, Python video pipeline, MediaPipe + YOLOv8 teacher extraction, and structured biomechanics exports.`

## What The App Can Do

Today, the desktop app can:

- accept a video by drag-and-drop, pasted path, or native `Choose File` picker
- let the user choose `Side View` or `Front Quarter`
- save a lightweight athlete profile from the upload screen
- run the janitor pipeline directly from the Rust app
- generate a visual preview snapshot from the uploaded clip
- detect one or more shots in the session
- show rep thumbnails for detected shots in the review selector
- extract and display a release-frame snapshot for the selected rep
- show coaching adjustments for the current shot
- show mechanical snapshot cards for extracted metrics
- show phase-by-phase feedback with `score / 100`
- draw a simplified visual overlay review panel
- print explicit model scores to terminal output while keeping the UI focused on visuals and coaching

## Project Structure

The repo has two main components:

- `janitor_python/`
  Video intake, metadata capture, teacher-model processing, shot segmentation, feature export, and corpus generation.
- `athlete_rust/`
  Native desktop UI plus the Rust-side mechanics review and lightweight scoring layer.

Supporting folders:

- `datasets/uploads/`
  Raw uploaded clips, manifests, processed sessions, temporary app snapshots, and tuning files.
- `datasets/shared/processed/`
  Shared training corpus exported as Parquet and metadata.
- `datasets/calibration_20_shot/`
  Calibration and validation data.
- `datasets/models/mediapipe/`
  Local MediaPipe model assets.
- `schemas/`
  Shared schema definitions.

## Current Pipeline

The current end-to-end flow looks like this:

1. A clip is selected in the Rust desktop app.
2. The app creates or updates an athlete profile JSON under `datasets/uploads/`.
3. The app calls the Python janitor CLI to ingest the video and write a manifest.
4. The janitor runs the stronger teacher pipeline with MediaPipe and YOLOv8-based extraction.
5. Shot records are exported and the shared training corpus is rebuilt.
6. Rust loads the processed shot records, analyzes the selected shot, and renders the review UI.
7. The app extracts visual snapshots from the clip for preview and release context.

## Extracted Mechanics

The structured shot features currently include:

- elbow flexion
- knee load
- forearm verticality
- elbow flare
- release height ratio
- release timing
- release vs. apex offset
- jump height

These features feed both the coaching view and the lightweight Rust-side supervised score layer.

## Desktop Review Experience

After analysis completes, the app currently shows:

- a large preview image from the uploaded clip
- shot label and rep count summary cards
- a rep selector with extracted thumbnails
- coaching cards under `What To Adjust`
- a `Mechanical Snapshot` section with the main extracted metrics
- a `Shot Phases` section with phase feedback and per-phase scores
- a release snapshot for the selected rep
- a simplified visual overlay panel for the motion pattern

The UI is intentionally focused on coaching and visuals. Explicit model scores are logged to standard output instead of being the main visual element on screen.

## Quick Start

### 1. Set Up Python

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/janitor_python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Make Sure Model Assets Exist

Expected local assets:

- `yolov8n.pt`
- `yolov8n-pose.pt`
- `datasets/models/mediapipe/pose_landmarker_lite.task`

### 3. Run The Desktop App

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/athlete_rust
cargo run
```

### 4. Analyze A Clip

Inside the app:

1. drag a video into the window, paste a full path, or click `Choose File`
2. select `Side View` or `Front Quarter`
3. fill in or confirm the athlete form values
4. click `Analyze Video`
5. review the generated snapshots, shot selector thumbnails, coaching cards, metrics, and phase scores

## CLI Workflow

If you want to run the janitor pipeline manually instead of through the desktop app, the main commands are:

### Intake A Clip

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor intake-video \
  --project-root . \
  --clip /absolute/path/to/clip.mp4 \
  --view side
```

Use `--view angle45` for front-quarter video.

### Process A Session

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor strong-process \
  --project-root . \
  --manifest datasets/uploads/manifests/<manifest>.json \
  --athlete-profile datasets/uploads/app_athlete.json \
  --source-dataset uploaded_session \
  --teacher-model mediapipe_yolov8_teacher \
  --frame-stride 1 \
  --yolo-weights yolov8n.pt \
  --pose-weights yolov8n-pose.pt \
  --mediapipe-model datasets/models/mediapipe/pose_landmarker_lite.task
```

### Rebuild The Shared Corpus

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor build-corpus --project-root .
```

This writes:

- `datasets/shared/processed/training_corpus.parquet`
- `datasets/shared/processed/training_corpus.metadata.json`

## What “Training” Means Here

This is not yet an end-to-end learned biomechanics model.

Right now the system is:

- teacher-driven for perception
- feature-driven for mechanics
- lightly supervised on the Rust side for structured-score prediction

In practice:

1. teacher models detect pose and ball signals
2. temporal and geometric logic convert those signals into shot features
3. Rust reads the processed features and fits or applies a lightweight score layer

That means the app is already practical for review and coaching, while the longer-term model story is still evolving.

## Troubleshooting

- If the desktop app finishes analysis but no shots appear, the janitor likely did not detect a usable rep from that clip.
- If visual snapshots do not appear, make sure the janitor Python environment exists and can import OpenCV.
- If the app shows stale processed data, rerun analysis or rebuild the shared corpus.
- If MediaPipe is unavailable on a machine, the janitor pipeline is designed to fall back rather than block the whole review flow.

## Status

Current product direction:

- native Rust desktop review
- Python-backed video ingestion and teacher extraction
- structured biomechanics exports
- visual shot review with snapshots and thumbnails
- coaching-focused UI instead of raw-score-first UI

The long-term goal is a stronger jump-shot training system that keeps the current practical upload-and-review workflow while improving model quality, pairing accuracy, and generalization across athletes and recording conditions.
