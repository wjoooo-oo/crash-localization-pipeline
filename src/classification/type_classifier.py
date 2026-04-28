"""
Crash Type Classification

RT-DETR detection + ByteTrack + Rules-based reranker
"""

from typing import List, Dict
from pathlib import Path
import pandas as pd


CRASH_TYPES = ["single", "t-bone", "rear-end", "sideswipe", "head-on"]


class CrashTypeClassifier:
    """Multi-stage crash type classifier"""

    def __init__(self, detector_model: str = "rtdetr-x"):
        """
        Initialize classifier.

        Args:
            detector_model: RT-DETR model variant
        """
        # Load RT-DETR
        # Initialize ByteTrack
        # Load scene correction rules
        pass

    def detect_objects(self, image_path: str) -> List[Dict]:
        """
        Run object detection on image.

        Returns:
            List of detections with bbox, class, confidence
        """
        raise NotImplementedError("See accident/test_metadata/")

    def track_objects(self, detections: List[Dict], video_path: str) -> List[Dict]:
        """
        Multi-object tracking with ByteTrack.

        Returns:
            List of tracks with trajectories
        """
        raise NotImplementedError()

    def infer_type_from_geometry(self, tracks: List[Dict]) -> str:
        """
        Infer crash type from object trajectories.

        Rules:
        - Rear-end: Sequential boxes, decreasing distance
        - Head-on: Opposing motion vectors
        - T-bone: Perpendicular trajectories
        - Sideswipe: Parallel motion, lateral contact
        - Single: One vehicle, no interaction
        """
        raise NotImplementedError()

    def apply_scene_corrections(self, predicted_type: str, scene_info: Dict) -> str:
        """
        Scene-aware reranking.

        SCENE_CORRECTIONS from:
        /home/disk3/Jiachen/accident/test_metadata/rerank_type_majority.py
        """
        raise NotImplementedError()

    def predict(self, video_path: str, onset_time: float, bbox: List[float]) -> str:
        """
        Predict crash type for a single video.

        Args:
            video_path: Path to video
            onset_time: Predicted onset time
            bbox: Spatial bounding box

        Returns:
            Crash type string
        """
        # Extract onset frame
        # Run detection
        # Track objects
        # Infer type from geometry
        # Apply scene corrections
        # Return final type
        raise NotImplementedError()


def run_classification(
    temporal_csv: str,
    spatial_csv: str,
    video_dir: str,
    output_csv: str
):
    """
    Run full classification pipeline.

    Full implementation:
    /home/disk3/Jiachen/accident/test_metadata/rerank_type_majority.py

    Args:
        temporal_csv: Predicted onset times
        spatial_csv: Predicted bounding boxes
        video_dir: Video directory
        output_csv: Output CSV path
    """
    classifier = CrashTypeClassifier()

    # Load predictions
    temporal_df = pd.read_csv(temporal_csv)
    spatial_df = pd.read_csv(spatial_csv)

    # Merge on video_path
    merged = temporal_df.merge(spatial_df, on="video_path")

    # Predict types
    results = []
    for _, row in merged.iterrows():
        crash_type = classifier.predict(
            row["video_path"],
            row["predicted_onset"],
            [row[f"bbox_{c}"] for c in ["x_min", "y_min", "x_max", "y_max"]]
        )
        results.append({
            "video_path": row["video_path"],
            "crash_type": crash_type
        })

    # Save
    pd.DataFrame(results).to_csv(output_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-csv", required=True)
    parser.add_argument("--spatial-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    run_classification(
        args.temporal_csv,
        args.spatial_csv,
        args.video_dir,
        args.output_csv
    )
