# JumpShot Trainer

JumpShot Trainer is a Rust-first basketball shooting analysis app with a Python ingestion pipeline.

It is designed to turn raw training clips into:

- shot-stage segmentation
- biomechanics features
- session audits
- trainable Parquet corpora
- a desktop review experience for solo shooting workouts

Suggested GitHub repository description:

`Rust biomechanics shooting trainer with a Python video-ingestion pipeline, MediaPipe + YOLOv8 teacher models, Parquet feature exports, and a desktop analysis dashboard.`

## What The App Does

The project is split into two parts:

- `janitor_python/`
  Handles raw video intake, clip metadata extraction, teacher-model processing, shot segmentation, feature extraction, and Parquet export.
- `athlete_rust/`
  Handles calibration, biomechanics scoring, dataset inspection, session review, and the desktop dashboard.

In practical terms:

1. You drop side-view and front-quarter videos into the upload inbox.
2. Python processes those videos into frame observations and shot records.
3. The processed records are exported into a shared training corpus.
4. Rust reads that corpus and turns it into a review dashboard, session summaries, and model-readiness signals.

## How The System Works

### 1. Video Intake

The Python janitor stores:

- raw clip copy
- manifest JSON
- FPS
- resolution
- duration
- orientation

This creates a stable source-of-truth archive under `datasets/uploads/`.

### 2. Teacher Extraction

The strong teacher path is:

- YOLOv8 for basketball detection
- MediaPipe pose when available
- automatic fallback to YOLO pose
- automatic fallback to the built-in pose heuristic if MediaPipe and YOLO pose are unavailable

This is intentionally defensive. On this macOS environment, MediaPipe task models can fail to initialize because of OpenGL service setup, so the pipeline automatically drops to YOLO pose instead of crashing the run.

### 3. Shot Segmentation

The janitor builds frame observations and then segments reps using:

- ball-to-hand distance when ball tracking is trustworthy
- wrist trajectory fallback when ball tracking is weak
- stage-window cleanup that removes impossible or unstable shot windows

The segmentation is still being tuned against real footage, but it is now running end-to-end on the uploaded sessions in this repo.

For difficult clips, the janitor also supports clip-specific session tuning with manual stage seeds. That path is now used for the weakest uploaded sessions in this repo.

### 4. Feature Extraction

For each detected shot, the janitor computes shot-level features such as:

- elbow flexion
- knee load
- forearm verticality
- elbow flare
- release height ratio
- release timing
- release vs apex offset
- jump height

These are written into Parquet so the Rust side can use real extracted mechanics instead of only demo values.

### 5. Pairing Side + Front Sessions

The corpus builder now attempts to merge same-day uploaded side/front sessions into paired records.

That gives the Rust app a better combined record where:

- side view contributes timing, load, and jump signals
- front-quarter view contributes alignment, flare, and release-line signals

This pairing is still heuristic. It now matches shots using same-day grouping plus normalized session progress and timing similarity, which is stronger than the original ordered-sequence-only pass.

### 6. Rust Review App

The Rust app reads the shared training corpus and shows:

- athlete calibration
- training readiness
- processed session coverage
- session and shot browser
- diagnostics
- session audit
- visual review panels
- heuristic and supervised Rust-side scores

The dashboard is no longer just generic scaffolding. It now reads real uploaded-session corpus rows and surfaces processed-session summaries directly.

## What “Trained” Means Right Now

This part matters.

The app is not yet a fully trained end-to-end vision system in the sense of:

- raw video in
- production-grade learned pose + ball + stage detector out
- fully supervised biomechanics model trained on a large labeled corpus

What it is today:

- a real teacher-driven ingestion pipeline
- a real feature-extraction pipeline
- a real Rust desktop app reading processed biomechanics records
- a Rust-side heuristic scorer plus a fitted supervised score model over structured features
- a corpus that can now include uploaded-session data, paired-view records, and manual gold-set data

So the current model stack is:

- teacher models produce detections and landmarks
- heuristics and geometry compute biomechanics features
- Rust consumes those features for scoring and review

This is the correct MVP path, because it lets the product become useful before a large supervised dataset exists.

## How It Was Trained

Today, the system is best described as:

- teacher-labeled and feature-driven
- not yet broadly supervised on a large custom basketball corpus

The current training story is:

1. Use teacher models to extract pose and ball signals from raw clips.
2. Convert those into biomechanics features.
3. Export them into Parquet.
4. Use Rust to summarize dataset quality, produce training examples, and prepare for future Candle-based model training.

The Rust side now includes a simple supervised fitted model over the structured feature corpus, with train and validation error surfaced in the UI. It is a meaningful step up from the original prototype scorer, but it is still not the final long-term biomechanics model.

The long-term training target is:

- a larger corpus of processed uploaded sessions
- gold validation data from manually tagged shots
- stronger paired-view coverage
- eventually a first true supervised Candle run on normalized feature tensors

## Current Repo State

Working today:

- raw upload intake
- per-clip manifest generation
- processed frame Parquet export
- processed shot-record Parquet export
- shared corpus rebuild
- uploaded-session corpus inclusion
- paired uploaded-session record generation
- Rust corpus ingestion
- Rust processed-session browser and dashboard summaries
- Rust supervised score model fitted from the current corpus

Currently validated in this repo:

- 4 uploaded videos were ingested and processed
- one side upload produced usable shot records
- one front upload produced a larger usable shot set
- the two originally weak uploaded sessions were rescued with clip-specific tuning and now each contribute usable shot records
- the shared corpus now includes manual-seeded rescue shots in addition to teacher-only shots
- multiple paired uploaded-session records are now created from same-day side/front sessions

Still in active tuning:

- shot count realism on long sessions
- release timing realism on sparse-stride runs
- stronger pairing quality between independent side and front recordings
- cleaner stage boundaries on difficult clips

## Folder Structure

- `athlete_rust/`
  Rust desktop application and analysis engine.
- `janitor_python/`
  Python ingestion and export pipeline.
- `datasets/uploads/`
  Raw session archive, manifests, processed sessions, and upload inbox.
- `datasets/calibration_20_shot/`
  Manual or semi-manual gold-tier calibration/validation data.
- `datasets/shared/processed/`
  Shared Parquet corpus read by Rust.
- `datasets/models/mediapipe/`
  Local MediaPipe pose landmarker task models.
- `schemas/`
  Shared schema contracts.
- `docs/`
  Product and architecture notes.

## How To Use It

### 1. Install Python Dependencies

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/janitor_python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Put Videos In The Upload Inbox

- side clips:
  `/Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/datasets/uploads/inbox/side`
- front or front-quarter clips:
  `/Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/datasets/uploads/inbox/front`

### 3. Intake Videos

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/janitor_python
source .venv/bin/activate
jumpshot-janitor intake-video --project-root .. --clip /absolute/path/to/clip.mp4 --view side
```

Use `--view angle45` for the front-quarter camera.

### 4. Add Local Teacher Assets

Recommended local assets:

- `yolov8n.pt`
- `yolov8n-pose.pt`
- `datasets/models/mediapipe/pose_landmarker_lite.task`

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

Outputs are written under:

- `datasets/uploads/processed/<session_id>/`

Optional weak-session rescue path:

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

The tuning file can provide manual shot seeds when the default segmenter still fails to recover usable rep windows.
Those rescue shots are preserved in the corpus with `has_manual_stage_tags=true`, and the Rust app now calls them out directly in the session browser.

### 6. Rebuild The Shared Corpus

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2
janitor_python/.venv/bin/jumpshot-janitor build-corpus --project-root .
```

Outputs:

- `datasets/shared/processed/training_corpus.parquet`
- `datasets/shared/processed/training_corpus.metadata.json`

### 7. Run The Rust App

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/athlete_rust
cargo run
```

### 8. Verify Rust Builds

```bash
cd /Users/ktr/Developer/GitHub/Jumpshot-Trainer-v2/athlete_rust
cargo check
```

## What The Rust App Shows

Calibration screen:

- body geometry inputs
- estimated wingspan
- estimated standing reach
- estimated camera angle
- dataset readiness summary

Dashboard:

- live shot intelligence
- supervised model summary
- training readiness
- processed session summaries
- session browser with per-shot drill-down
- diagnostics
- shot audit
- kinetic-chain stages
- visual review panel

## Recommended Next Steps

1. Keep processing real uploaded clips and tune stage thresholds against actual shot rhythm.
2. Capture more matched side/front sessions so pairing quality improves.
3. Replace heuristic pairing with true multi-view synchronization.
4. Add stronger basketball-specific ball detection weights.
5. Train the first true Rust-side supervised model on the structured feature corpus.

## Validation

Verified recently:

- Python janitor modules compile cleanly
- Rust app passes `cargo check`
- all 4 uploaded videos were ingested and processed into session artifacts
- the two weakest uploaded sessions were rescued with manual stage seeding
- the shared corpus now has 44 rows, including uploaded-session rows, paired uploaded-session records, and manual-stage-tagged rescue shots
