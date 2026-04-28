#!/usr/bin/env python3
"""
Evaluate predictions on 200-GT subset

Computes T, S, C scores and harmonic mean.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes [x_min, y_min, x_max, y_max]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def temporal_score(pred_time, gt_time, threshold=30, fps=30):
    """Temporal accuracy: 1.0 if within threshold frames, else 0.0"""
    frame_diff = abs(pred_time * fps - gt_time * fps)
    return 1.0 if frame_diff < threshold else 0.0


def evaluate_submission(pred_csv, gt_csv):
    """
    Evaluate predictions against ground truth.

    Returns:
        dict with T, S, C scores and harmonic mean
    """
    # Load CSVs
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    # Merge on video_id
    merged = pred_df.merge(gt_df, on="video_id", suffixes=("_pred", "_gt"))

    # Compute T scores
    t_scores = []
    for _, row in merged.iterrows():
        score = temporal_score(row["onset_time_pred"], row["onset_time_gt"])
        t_scores.append(score)

    # Compute S scores (IoU)
    s_scores = []
    for _, row in merged.iterrows():
        pred_bbox = [
            row["bbox_x_min_pred"], row["bbox_y_min_pred"],
            row["bbox_x_max_pred"], row["bbox_y_max_pred"]
        ]
        gt_bbox = [
            row["bbox_x_min_gt"], row["bbox_y_min_gt"],
            row["bbox_x_max_gt"], row["bbox_y_max_gt"]
        ]
        iou = compute_iou(pred_bbox, gt_bbox)
        s_scores.append(iou)

    # Compute C scores (exact match)
    c_scores = []
    for _, row in merged.iterrows():
        match = 1.0 if row["crash_type_pred"] == row["crash_type_gt"] else 0.0
        c_scores.append(match)

    # Average scores
    T = np.mean(t_scores)
    S = np.mean(s_scores)
    C = np.mean(c_scores)

    # Harmonic mean
    HM = 3 / (1/T + 1/S + 1/C) if (T > 0 and S > 0 and C > 0) else 0.0

    return {
        "T": T,
        "S": S,
        "C": C,
        "Harmonic_Mean": HM,
        "num_videos": len(merged)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate crash localization predictions")
    parser.add_argument("--pred-csv", required=True, help="Path to predictions CSV")
    parser.add_argument("--gt-csv", required=True, help="Path to ground truth CSV")
    args = parser.parse_args()

    # Validate files exist
    if not Path(args.pred_csv).exists():
        print(f"ERROR: Predictions file not found: {args.pred_csv}")
        return 1

    if not Path(args.gt_csv).exists():
        print(f"ERROR: Ground truth file not found: {args.gt_csv}")
        return 1

    # Evaluate
    print(f"Evaluating: {args.pred_csv}")
    print(f"Ground truth: {args.gt_csv}")
    print()

    scores = evaluate_submission(args.pred_csv, args.gt_csv)

    # Print results
    print("=" * 50)
    print(f"Evaluation Results ({scores['num_videos']} videos)")
    print("=" * 50)
    print(f"T (Temporal):        {scores['T']:.4f}")
    print(f"S (Spatial):         {scores['S']:.4f}")
    print(f"C (Classification):  {scores['C']:.4f}")
    print("-" * 50)
    print(f"Harmonic Mean:       {scores['Harmonic_Mean']:.4f}")
    print("=" * 50)

    # Compare to Sub5 baseline
    sub5_hm = 0.5391
    diff = scores['Harmonic_Mean'] - sub5_hm
    print()
    print(f"Sub5 Baseline HM: {sub5_hm:.4f}")
    print(f"Difference:       {diff:+.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
