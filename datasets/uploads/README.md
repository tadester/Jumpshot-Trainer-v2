# Upload Sessions

This area supports the raw-video intake flow.

- `inbox/side/`
  Drop raw side-view session videos here before intake.
- `inbox/front/`
  Drop raw front or front-quarter session videos here before intake.
- `raw/<session_id>/`
  Stored uploaded source clips.
- `manifests/`
  Per-clip metadata extracted during intake.
- `processed/<session_id>/`
  Frame observations, shot records, and processed session summaries.

The Python janitor can:

1. intake a raw clip and extract fps/resolution/duration/orientation
2. process a session using teacher-model pose/ball JSON outputs
3. export shot-level Parquet that rolls into the shared training corpus

Notes:

- `front/` is acceptable for now and is treated as `angle45` during processing if you intake it that way.
- Raw inbox videos, copied upload sources, manifests, and processed sessions are intentionally ignored by git.
- For the strongest teacher path, add a MediaPipe pose landmarker `.task` model under `datasets/models/mediapipe/`.
