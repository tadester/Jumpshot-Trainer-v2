# Upload Sessions

This area supports the raw-video intake flow.

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
