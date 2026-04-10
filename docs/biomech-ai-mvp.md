# BioMech-AI MVP Architecture

## Product Goal

Build a real basketball form-analysis MVP that can:

- take uploaded shooting clips
- estimate pose and ball motion
- segment reps
- compute biomechanics features
- store those features in a trainable corpus
- present the results in a Rust desktop review tool

## System Overview

The product has two execution layers.

### Janitor Python

Purpose:

- raw clip intake
- video metadata extraction
- teacher-model inference
- shot segmentation
- feature extraction
- corpus export

Primary outputs:

- per-session frame observations
- per-session shot records
- shared training corpus

### Athlete Rust

Purpose:

- calibration and normalization
- shot scoring
- dataset inspection
- desktop review UI
- readiness tracking for future training

Primary outputs:

- calibration deck
- diagnostics dashboard
- processed-session overview
- session audit and review surfaces

## Runtime Flow

1. User records side and front-quarter clips.
2. Python ingests the clips and writes manifests.
3. Teacher models estimate pose and ball behavior.
4. Python converts detections into frame-level observations.
5. Shot segmentation generates stage windows.
6. Feature extraction creates shot-level biomechanics records.
7. Corpus export writes Parquet.
8. Rust loads the corpus and renders review and readiness UI.

## Current Teacher Stack

Intended strong teacher:

- MediaPipe pose landmarker
- YOLOv8 basketball detector

Current resilient stack:

1. MediaPipe pose + YOLOv8 ball when MediaPipe initializes
2. YOLO pose + YOLOv8 ball when MediaPipe fails
3. built-in pose heuristic + YOLOv8/OpenCV ball fallback when needed

This fallback ladder is important because it keeps the ingestion pipeline usable on imperfect environments instead of turning every teacher failure into a hard stop.

## Shot Segmentation Strategy

The segmentation layer currently mixes:

- ball-hand distance logic
- wrist trajectory logic
- cleanup rules for impossible stage orderings

This is not yet a final production shot-state model, but it is materially better than a simple trigger-only approach and is now able to produce usable shot records from uploaded sessions.

## Feature Set

The current shot record includes:

- elbow flexion
- knee load
- forearm verticality
- elbow flare
- release height ratio
- release timing
- release vs apex offset
- jump height

These features are the bridge between teacher-generated detections and future Rust-side supervised learning.

## Pairing Strategy

Uploaded sessions are now paired heuristically when:

- athlete id matches
- session date matches
- side and front-quarter records both exist

The current pairing method matches shots by ordered sequence within each same-day group. This is a practical MVP solution, but it should eventually be replaced by true multi-view synchronization.

## Training Positioning

The system is not yet a fully trained end-to-end biomechanics model.

What it is today:

- teacher-driven perception
- feature-driven biomechanics extraction
- a trainable corpus builder
- a Rust-side model-readiness, review, and supervised score layer

What remains for a real supervised model:

- more paired sessions
- more manually trusted gold validation data
- better shot-stage ground truth
- a first Candle-based supervised training pass

## What The Rust App Now Surfaces

- calibration geometry
- live shot controls and diagnostics
- dataset readiness
- processed session summaries
- paired-view coverage
- session audit panels
- visual review placeholders for timeline/overlay work

## Current MVP Status

Already working:

- upload intake
- manifest generation
- session processing
- corpus rebuild
- Rust corpus ingestion
- processed-session dashboard summaries
- processed-shot browser in the Rust UI
- a fitted supervised Rust-side score model over the current feature corpus

Still being tuned:

- stage realism on long sessions
- sparse-stride timing accuracy
- view pairing fidelity
- weak-session recovery for uploads that still process into zero-shot session shells
- final training labels and large-scale supervised fitting

## Recommended Next Build Steps

1. Improve segmentation thresholds using more real uploaded sessions.
2. Add true paired-session synchronization instead of sequence-based pairing.
3. Capture more side/front-quarter gold validation clips.
4. Expand the corpus before the first serious Candle training run.
5. Replace the remaining heuristic stage cleanup with a stronger temporal model.
