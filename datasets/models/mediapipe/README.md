# MediaPipe Models

Place MediaPipe pose landmarker task models in this folder.

Supported filenames:

- `pose_landmarker_heavy.task`
- `pose_landmarker_full.task`
- `pose_landmarker_lite.task`

The Python janitor will look here automatically when running `strong-process`.

If no MediaPipe `.task` model is present, the pipeline will:

1. try YOLO pose weights if they are available locally
2. fall back to the built-in OpenCV pose heuristic if pose weights are missing

Recommended usage:

- keep YOLO ball weights in the repo root or pass a path with `--yolo-weights`
- keep a MediaPipe pose task model here or pass a path with `--mediapipe-model`
