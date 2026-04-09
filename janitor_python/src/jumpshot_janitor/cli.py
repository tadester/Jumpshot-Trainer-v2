from __future__ import annotations

import argparse
from pathlib import Path

from .exporters import export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Jumpshot janitor pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-parquet", help="Build shot-record Parquet from annotations")
    build.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    build.add_argument("--dataset", type=str, default="calibration_20_shot", help="Dataset folder name")

    args = parser.parse_args()

    if args.command == "build-parquet":
        dataset_parquet, shared_parquet = export_dataset(args.project_root.resolve(), args.dataset)
        print(f"Wrote dataset parquet: {dataset_parquet}")
        print(f"Wrote shared parquet: {shared_parquet}")
