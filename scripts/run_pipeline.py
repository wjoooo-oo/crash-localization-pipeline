#!/usr/bin/env python3
"""
Full Crash Localization Pipeline

Runs T → S → C stages sequentially and generates submission CSV.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal.inference import run_temporal_inference
from spatial.vlm_predictor import run_spatial_inference
from classification.type_classifier import run_classification


def merge_predictions(temporal_csv: str, spatial_csv: str, type_csv: str, output_csv: str):
    """
    Merge T, S, C predictions into final submission format.

    Final CSV columns:
    - video_id: str
    - onset_time: float (seconds)
    - bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max: float [0,1]
    - crash_type: str
    """
    # Load all predictions
    df_t = pd.read_csv(temporal_csv)
    df_s = pd.read_csv(spatial_csv)
    df_c = pd.read_csv(type_csv)

    # Merge on video_path
    merged = df_t.merge(df_s, on="video_path").merge(df_c, on="video_path")

    # Extract video_id from path
    merged["video_id"] = merged["video_path"].apply(lambda x: Path(x).stem)

    # Rename columns to submission format
    submission = merged[[
        "video_id",
        "predicted_onset",
        "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
        "crash_type"
    ]].rename(columns={"predicted_onset": "onset_time"})

    # Save
    submission.to_csv(output_csv, index=False)
    print(f"✓ Submission saved to {output_csv} ({len(submission)} videos)")


def main():
    parser = argparse.ArgumentParser(description="Run full crash localization pipeline")
    parser.add_argument("--video-dir", required=True, help="Directory containing test videos")
    parser.add_argument("--temporal-checkpoint", required=True, help="Path to VideoMAE checkpoint")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--batch-size-vlm", type=int, default=4, help="VLM batch size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Stage 1: Temporal Prediction
    print("\n=== Stage 1: Temporal Prediction ===")
    temporal_csv = output_dir / "temporal_predictions.csv"
    run_temporal_inference(
        checkpoint_path=args.temporal_checkpoint,
        video_dir=args.video_dir,
        output_csv=str(temporal_csv)
    )
    print(f"✓ Temporal predictions: {temporal_csv}")

    # Stage 2: Spatial Localization
    print("\n=== Stage 2: Spatial Localization ===")
    spatial_csv = output_dir / "spatial_predictions.csv"
    run_spatial_inference(
        onset_csv=str(temporal_csv),
        video_dir=args.video_dir,
        output_csv=str(spatial_csv),
        batch_size=args.batch_size_vlm
    )
    print(f"✓ Spatial predictions: {spatial_csv}")

    # Stage 3: Type Classification
    print("\n=== Stage 3: Type Classification ===")
    type_csv = output_dir / "type_predictions.csv"
    run_classification(
        temporal_csv=str(temporal_csv),
        spatial_csv=str(spatial_csv),
        video_dir=args.video_dir,
        output_csv=str(type_csv)
    )
    print(f"✓ Type predictions: {type_csv}")

    # Merge into final submission
    print("\n=== Generating Submission ===")
    submission_csv = output_dir / "submission.csv"
    merge_predictions(
        str(temporal_csv),
        str(spatial_csv),
        str(type_csv),
        str(submission_csv)
    )

    print("\n=== Pipeline Complete ===")
    print(f"Final submission: {submission_csv}")
    print("Next: Validate with evaluation/analyze_errors.py")


if __name__ == "__main__":
    main()
