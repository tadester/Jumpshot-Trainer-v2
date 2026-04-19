import pandas as pd
from src.jumpshot_janitor.schema import VideoManifest
from src.jumpshot_janitor.video_pipeline import _refine_shot_windows, _segment_shots_fallback_global

frames = []
for i in range(10):
    frames.append({
        "frame_index": i,
        "timestamp_ms": i * 33.3,
        "wrist_y": 100.0 if i != 5 else 50.0,
        "hip_y": 200.0 if i != 5 else 150.0
    })

df = pd.DataFrame(frames)
shots = _segment_shots_fallback_global(df)
refined = _refine_shot_windows(shots, df)

manifest = VideoManifest(
    session_id="dummy",
    clip_id="dummy",
    view="side",
    fps=60.0,
    stored_path="",
)

features = _extract_shot_features(
    refined, df, manifest, {"athlete_id": "test", "height_m": 1.9, "standing_reach_m": 2.5}, "test", "test"
)

print("Features:")
print(features)
