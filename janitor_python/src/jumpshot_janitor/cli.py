from __future__ import annotations

import argparse
from pathlib import Path

from .exporters import export_dataset, export_training_corpus
from .video_pipeline import auto_process_session, auto_process_session_strong, intake_video, process_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Jumpshot janitor pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-parquet", help="Build shot-record Parquet from annotations")
    build.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    build.add_argument("--dataset", type=str, default="calibration_20_shot", help="Dataset folder name")
    corpus = subparsers.add_parser("build-corpus", help="Build a shared multi-dataset training corpus")
    corpus.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    intake = subparsers.add_parser("intake-video", help="Store a raw uploaded clip and extract video metadata")
    intake.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    intake.add_argument("--clip", type=Path, required=True, help="Path to the uploaded raw clip")
    intake.add_argument("--view", choices=["side", "angle45"], required=True, help="Camera view for the clip")
    process = subparsers.add_parser("process-session", help="Process an uploaded session with teacher-model outputs")
    process.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    process.add_argument("--manifest", type=Path, required=True, help="Path to the intake manifest json")
    process.add_argument("--athlete-profile", type=Path, required=True, help="Athlete profile json used for normalization")
    process.add_argument("--pose-json", type=Path, required=True, help="Teacher-model pose output json")
    process.add_argument("--ball-json", type=Path, help="Teacher-model ball-track output json")
    process.add_argument("--source-dataset", type=str, default="uploaded_session", help="Logical dataset/source name")
    process.add_argument("--teacher-model", type=str, default="teacher_import", help="Teacher model identifier")
    auto = subparsers.add_parser("auto-process", help="Run the built-in CV teacher and process a raw uploaded session")
    auto.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    auto.add_argument("--manifest", type=Path, required=True, help="Path to the intake manifest json")
    auto.add_argument("--athlete-profile", type=Path, required=True, help="Athlete profile json used for normalization")
    auto.add_argument("--source-dataset", type=str, default="uploaded_session", help="Logical dataset/source name")
    auto.add_argument("--teacher-model", type=str, default="builtin_cv_teacher", help="Teacher model identifier")
    auto.add_argument("--frame-stride", type=int, default=2, help="Analyze every Nth frame")
    strong = subparsers.add_parser("strong-process", help="Run MediaPipe pose + YOLOv8 ball teacher on a raw uploaded session")
    strong.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    strong.add_argument("--manifest", type=Path, required=True, help="Path to the intake manifest json")
    strong.add_argument("--athlete-profile", type=Path, required=True, help="Athlete profile json used for normalization")
    strong.add_argument("--source-dataset", type=str, default="uploaded_session", help="Logical dataset/source name")
    strong.add_argument("--teacher-model", type=str, default="mediapipe_yolov8_teacher", help="Teacher model identifier")
    strong.add_argument("--frame-stride", type=int, default=2, help="Analyze every Nth frame")
    strong.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLOv8 weights name or path")
    strong.add_argument("--pose-weights", type=str, default="yolov8n-pose.pt", help="YOLO pose weights used if MediaPipe is unavailable")
    strong.add_argument("--mediapipe-model", type=str, help="Optional path to a MediaPipe pose landmarker .task model")

    args = parser.parse_args()

    if args.command == "build-parquet":
        dataset_parquet, shared_parquet = export_dataset(args.project_root.resolve(), args.dataset)
        print(f"Wrote dataset parquet: {dataset_parquet}")
        print(f"Wrote shared parquet: {shared_parquet}")
    if args.command == "build-corpus":
        corpus_parquet = export_training_corpus(args.project_root.resolve())
        print(f"Wrote training corpus: {corpus_parquet}")
    if args.command == "intake-video":
        manifest_path = intake_video(args.project_root.resolve(), args.clip.resolve(), args.view)
        print(f"Wrote intake manifest: {manifest_path}")
    if args.command == "process-session":
        outputs = process_session(
            project_root=args.project_root.resolve(),
            manifest_path=args.manifest.resolve(),
            athlete_profile_path=args.athlete_profile.resolve(),
            pose_json=args.pose_json.resolve(),
            ball_json=args.ball_json.resolve() if args.ball_json else None,
            source_dataset=args.source_dataset,
            teacher_model=args.teacher_model,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
    if args.command == "auto-process":
        outputs = auto_process_session(
            project_root=args.project_root.resolve(),
            manifest_path=args.manifest.resolve(),
            athlete_profile_path=args.athlete_profile.resolve(),
            source_dataset=args.source_dataset,
            teacher_model=args.teacher_model,
            frame_stride=args.frame_stride,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
    if args.command == "strong-process":
        outputs = auto_process_session_strong(
            project_root=args.project_root.resolve(),
            manifest_path=args.manifest.resolve(),
            athlete_profile_path=args.athlete_profile.resolve(),
            source_dataset=args.source_dataset,
            teacher_model=args.teacher_model,
            frame_stride=args.frame_stride,
            yolo_weights=args.yolo_weights,
            pose_weights=args.pose_weights,
            mediapipe_model=args.mediapipe_model,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
