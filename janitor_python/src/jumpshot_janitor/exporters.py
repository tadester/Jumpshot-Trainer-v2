from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

import pandas as pd

from .schema import AthleteProfile


REQUIRED_COLUMNS = [
    "shot_id",
    "session_date",
    "fps",
    "side_video",
    "angle45_video",
    "set_point_frame_side",
    "release_frame_side",
    "set_point_frame_45",
    "release_frame_45",
    "make",
    "shot_type",
    "distance_ft",
    "notes",
]


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[REQUIRED_COLUMNS].copy()


def _derive_columns(frame: pd.DataFrame, athlete: AthleteProfile) -> pd.DataFrame:
    output = frame.copy()
    output["athlete_id"] = athlete.athlete_id
    output["handedness"] = athlete.handedness
    output["height_m"] = athlete.height_m
    output["wingspan_m"] = athlete.wingspan_m
    output["standing_reach_m"] = athlete.standing_reach_m

    fps = pd.to_numeric(output["fps"], errors="coerce").fillna(athlete.capture_fps or 60)
    output["fps"] = fps.astype("Int64")

    for column in [
        "set_point_frame_side",
        "release_frame_side",
        "set_point_frame_45",
        "release_frame_45",
        "distance_ft",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output["release_time_ms_side"] = (
        (output["release_frame_side"] - output["set_point_frame_side"]) / output["fps"] * 1000.0
    )
    output["release_time_ms_45"] = (
        (output["release_frame_45"] - output["set_point_frame_45"]) / output["fps"] * 1000.0
    )
    output["paired_view_available"] = (
        output["side_video"].fillna("").astype(str).str.len().gt(0)
        & output["angle45_video"].fillna("").astype(str).str.len().gt(0)
    )
    return output


def build_shot_records(project_root: Path, dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = project_root / "datasets" / dataset_name
    annotations_root = dataset_root / "annotations"
    athlete = AthleteProfile.load(annotations_root / "athlete_profile.json")
    shots = pd.read_csv(annotations_root / "shots.csv")
    shots = _ensure_columns(shots)
    records = _derive_columns(shots, athlete)

    metadata = {
        "dataset_name": dataset_name,
        "athlete_profile": athlete.to_dict(),
        "row_count": int(len(records)),
        "columns": list(records.columns),
    }
    return records, metadata


def export_dataset(project_root: Path, dataset_name: str) -> tuple[Path, Path]:
    records, metadata = build_shot_records(project_root, dataset_name)
    derived_root = project_root / "datasets" / dataset_name / "derived"
    shared_root = project_root / "datasets" / "shared" / "processed"
    derived_root.mkdir(parents=True, exist_ok=True)
    shared_root.mkdir(parents=True, exist_ok=True)

    dataset_parquet = derived_root / "shot_records.parquet"
    shared_parquet = shared_root / f"{dataset_name}_shot_records.parquet"
    metadata_json = derived_root / "shot_records.metadata.json"

    records.to_parquet(dataset_parquet, index=False)
    records.to_parquet(shared_parquet, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2))
    return dataset_parquet, shared_parquet
