# BioMech-AI MVP Architecture

## Product Goal

Ship an MVP that converts pose landmarks and ball tracking into stage-aware biomechanical diagnostics, instant rep feedback, and trendable athlete progress.

## Core System

1. Capture Layer
   - Record 60-120 FPS video from a tripod or live camera session.
   - Run pose estimation and ball tracking per frame.
   - Timestamp all frames for latency, release timing, and jump-apex calculations.

2. Reference Layer
   - Store elite baseline ranges in `data/elite-shot-baselines.json`.
   - Validate model robustness using `data/environment-benchmarks.json`.
   - Replace placeholder values with empirically labeled pro-shooter clips.

3. Analysis Layer
   - Detect shot stages with a Rust state machine: `ready_stance -> load -> set_point -> release -> follow_through`.
   - Compute proportion-aware metrics including elbow flexion, knee load, forearm verticality, elbow flare, release height ratio, and release timing.
   - Generate severity-based diagnostics for real-time overlays and post-shot audits.

4. Product Layer
   - Feed diagnostics into AR overlays, angle meters, and session summaries.
   - Persist normalized shot records for athlete history.
   - Clip each rep to a short shot window for efficient storage and review.

## MVP Deliverables Mapping

- Ground truth:
  `data/elite-shot-baselines.json` defines the first-pass golden-ratio baseline schema.
- Multi-stage recognition:
  `src/analysis/state_machine.rs` detects the shot phases.
- Biomechanical diagnostics:
  `src/analysis/diagnostics.rs` computes body-relative metrics and compares them to the pro baseline.
- Session summary:
  `src/analysis/session_audit.rs` ranks consistency across attempts.
- Persistence and trend analysis:
  `src/backend/persistence.rs` normalizes scores and computes improvement windows.
- Automated clipping:
  `src/backend/video_window.rs` identifies a compact shot window.

## Portfolio-Ready Outputs

- Diagnostic report dashboard:
  Back the report with `MetricDiagnostic[]`, including statements like elbow flare deviation versus baseline.
- Frame-by-frame scrubber:
  Bind shot stage events and metric snapshots to a timeline view.
- Latency audit:
  Track capture FPS, inference time per frame, and total end-to-end lag in the ingestion service.

## Recommended Next Build Steps

1. Connect a pose stack such as MediaPipe Pose Landmarker or MoveNet plus a ball detector.
2. Capture 120 FPS footage of 10-20 shooters and replace the synthetic baseline values with measured means and tolerances.
3. Add a calibration routine to estimate wingspan, standing reach, and camera angle before the first session.
4. Build a dashboard or mobile review screen that renders shot stages, color-coded overlays, and session audits.
