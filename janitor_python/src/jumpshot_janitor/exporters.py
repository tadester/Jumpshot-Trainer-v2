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
    "shot_start_frame_side",
    "set_point_frame_side",
    "release_frame_side",
    "shot_end_frame_side",
    "shot_start_frame_45",
    "set_point_frame_45",
    "release_frame_45",
    "shot_end_frame_45",
    "make",
    "shot_type",
    "distance_ft",
    "source_dataset",
    "source_tier",
    "annotation_quality",
    "teacher_model",
    "clip_uid",
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
    output["source_dataset"] = output["source_dataset"].fillna(athlete.athlete_id)
    output["source_tier"] = output["source_tier"].fillna(athlete.source_tier or "gold")
    output["annotation_quality"] = output["annotation_quality"].fillna("manual")
    output["teacher_model"] = output["teacher_model"].fillna("manual_review")
    output["clip_uid"] = output["clip_uid"].fillna(output["shot_id"])

    fps = pd.to_numeric(output["fps"], errors="coerce").fillna(athlete.capture_fps or 60)
    output["fps"] = fps.astype("Int64")

    for column in [
        "shot_start_frame_side",
        "set_point_frame_side",
        "release_frame_side",
        "shot_end_frame_side",
        "shot_start_frame_45",
        "set_point_frame_45",
        "release_frame_45",
        "shot_end_frame_45",
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
    output["release_timing_gap_ms"] = (
        output["release_time_ms_side"].fillna(output["release_time_ms_45"])
        - output["release_time_ms_45"].fillna(output["release_time_ms_side"])
    ).abs()
    output["has_manual_stage_tags"] = (
        output["set_point_frame_side"].notna() | output["set_point_frame_45"].notna()
    ) & (
        output["release_frame_side"].notna() | output["release_frame_45"].notna()
    )
    output["is_training_candidate"] = (
        output["has_manual_stage_tags"]
        | output["annotation_quality"].astype(str).isin(["teacher", "hybrid", "manual"])
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


def build_training_corpus(project_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = project_root / "datasets"
    records_collection: list[pd.DataFrame] = []
    dataset_names: list[str] = []

    for candidate in sorted(dataset_root.iterdir()):
        if not candidate.is_dir() or candidate.name == "shared":
            continue
        annotations_root = candidate / "annotations"
        if not (annotations_root / "athlete_profile.json").exists() or not (annotations_root / "shots.csv").exists():
            continue
        dataset_names.append(candidate.name)
        athlete = AthleteProfile.load(annotations_root / "athlete_profile.json")
        shots = _ensure_columns(pd.read_csv(annotations_root / "shots.csv"))
        records_collection.append(_derive_columns(shots, athlete))

    uploads_processed = dataset_root / "uploads" / "processed"
    if uploads_processed.exists():
        for parquet_path in sorted(uploads_processed.glob("*/*_shot_records.parquet")):
            dataset_names.append(f"processed:{parquet_path.parent.name}")
            records_collection.append(pd.read_parquet(parquet_path))

    corpus = pd.concat(records_collection, ignore_index=True) if records_collection else pd.DataFrame(columns=REQUIRED_COLUMNS)
    metadata = {
        "dataset_names": dataset_names,
        "row_count": int(len(corpus)),
        "columns": list(corpus.columns),
    }
    return corpus, metadata


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


def export_training_corpus(project_root: Path) -> Path:
    corpus, metadata = build_training_corpus(project_root)
    shared_root = project_root / "datasets" / "shared" / "processed"
    shared_root.mkdir(parents=True, exist_ok=True)
    corpus_parquet = shared_root / "training_corpus.parquet"
    metadata_json = shared_root / "training_corpus.metadata.json"
    corpus.to_parquet(corpus_parquet, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2))
    return corpus_parquet
