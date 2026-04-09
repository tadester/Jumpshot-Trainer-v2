# BioMech-AI MVP Architecture

## Product Goal

Ship an MVP that converts raw uploaded basketball clips into stage-aware biomechanical diagnostics, instant rep feedback, and a trainable long-term shot corpus.

## Core System

1. Capture Layer
   - Record 60-120 FPS video from a tripod or live camera session.
   - Intake raw video into the Python janitor and store a manifest with FPS, resolution, duration, and orientation.
   - Run teacher-model inference per frame for pose and ball tracking.
   - Timestamp all frames for latency, release timing, and jump-apex calculations.

2. Reference Layer
   - Store elite baseline ranges in `athlete_rust/data/elite-shot-baselines.json`.
   - Validate model robustness using `athlete_rust/data/environment-benchmarks.json`.
   - Replace placeholder values with empirically labeled pro-shooter clips and teacher-processed corpora.

3. Analysis Layer
   - Detect shot stages with a Rust state machine: `ready_stance -> load -> set_point -> release -> follow_through`.
   - Compute proportion-aware metrics including elbow flexion, knee load, forearm verticality, elbow flare, release height ratio, and release timing.
   - Generate severity-based diagnostics for real-time overlays and post-shot audits.
   - Convert processed shot records into normalized training examples for Rust-side training.

4. Product Layer
   - Feed diagnostics into AR overlays, angle meters, and session summaries.
   - Persist normalized shot records for athlete history.
   - Clip each rep to a short shot window for efficient storage and review.
   - Surface corpus readiness so the team knows when to move from collection into supervised training.

## Implemented System Split

### Janitor Python

- raw video intake and manifest creation
- teacher-model processing paths
- frame-observation and shot-record export
- calibration-set export and shared training corpus export

### Athlete Rust

- calibration and normalization logic
- biomechanics scoring and session audit logic
- desktop review UI
- Parquet ingestion for processed corpora
- training-readiness summarization

## Teacher Backends

1. Strong teacher
   - MediaPipe pose
   - YOLOv8 basketball detection
2. Fallback teacher
   - OpenCV people detector
   - heuristic orange-ball detector

The strong teacher path should be the default for any meaningful training data collection.

## MVP Deliverables Mapping

- Ground truth:
  `athlete_rust/data/elite-shot-baselines.json` defines the first-pass golden-ratio baseline schema.
- Multi-stage recognition:
  `athlete_rust/src/analysis/state_machine.rs` detects the shot phases.
- Biomechanical diagnostics:
  `athlete_rust/src/analysis/diagnostics.rs` computes body-relative metrics and compares them to the pro baseline.
- Session summary:
  `athlete_rust/src/analysis/session_audit.rs` ranks consistency across attempts.
- Persistence and trend analysis:
  `athlete_rust/src/backend/persistence.rs` normalizes scores and computes improvement windows.
- Automated clipping:
  `athlete_rust/src/backend/video_window.rs` identifies a compact shot window.
- Video intake and teacher processing:
  `janitor_python/src/jumpshot_janitor/video_pipeline.py` handles raw uploads, teacher outputs, segmentation, and processed session export.
- Shared corpus:
  `datasets/shared/processed/training_corpus.parquet` is the Rust-readable corpus assembled from calibration sets and processed uploaded sessions.

## Portfolio-Ready Outputs

- Diagnostic report dashboard:
  Back the report with `MetricDiagnostic[]`, including statements like elbow flare deviation versus baseline.
- Frame-by-frame scrubber:
  Bind shot stage events and metric snapshots to a timeline view.
- Latency audit:
  Track capture FPS, inference time per frame, and total end-to-end lag in the ingestion service.

## Recommended Next Build Steps

1. Replace the heuristic fallback pieces with stronger production-grade teacher settings and better basketball-specific YOLO weights.
2. Capture 60-120 FPS footage of more shooters and keep the 20-shot set as gold-tier validation.
3. Export richer frame-level landmarks and confidence traces into the shared training corpus.
4. Train the first Rust-side supervised model on top of the processed corpus and compare it to the current heuristic scorer.
