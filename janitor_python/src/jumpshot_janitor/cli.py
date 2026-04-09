from __future__ import annotations

import argparse
from pathlib import Path

from .exporters import export_dataset, export_training_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Jumpshot janitor pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-parquet", help="Build shot-record Parquet from annotations")
    build.add_argument("--project-root", type=Path, required=True, help="Repo root path")
    build.add_argument("--dataset", type=str, default="calibration_20_shot", help="Dataset folder name")
    corpus = subparsers.add_parser("build-corpus", help="Build a shared multi-dataset training corpus")
    corpus.add_argument("--project-root", type=Path, required=True, help="Repo root path")

    args = parser.parse_args()

    if args.command == "build-parquet":
        dataset_parquet, shared_parquet = export_dataset(args.project_root.resolve(), args.dataset)
        print(f"Wrote dataset parquet: {dataset_parquet}")
        print(f"Wrote shared parquet: {shared_parquet}")
    if args.command == "build-corpus":
        corpus_parquet = export_training_corpus(args.project_root.resolve())
        print(f"Wrote training corpus: {corpus_parquet}")
