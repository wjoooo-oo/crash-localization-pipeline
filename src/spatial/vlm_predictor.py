"""
Spatial Localization with Qwen VLM

Uses Qwen 3.5 27B VLM with 4-bit quantization to predict bounding boxes.
"""

from typing import Tuple, Dict
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json


class QwenSpatialPredictor:
    """Qwen VLM for spatial crash localization"""

    PROMPT_TEMPLATE = """
Analyze this traffic crash scene carefully.

Task: Locate the primary crash impact zone in the image.

Return a JSON object with:
{{
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": <float 0-1>,
  "reasoning": "<brief explanation>"
}}

Coordinates must be normalized to [0, 1] range.
"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-27B-Instruct", device: str = "cuda"):
        """
        Initialize Qwen VLM predictor.

        Args:
            model_name: Hugging Face model name
            device: Device to run on
        """
        self.device = device

        # Load with 4-bit quantization
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        Predict bounding box for a single image.

        Args:
            image_path: Path to onset frame image

        Returns:
            Dictionary with bbox, confidence, reasoning
        """
        image = Image.open(image_path).convert("RGB")

        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.PROMPT_TEMPLATE}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Parse JSON response
        try:
            result = json.loads(generated_text)
            return result
        except json.JSONDecodeError:
            print(f"Failed to parse VLM output: {generated_text}")
            return {
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "confidence": 0.0,
                "reasoning": "Parse error"
            }


def run_spatial_inference(
    onset_csv: str,
    video_dir: str,
    output_csv: str,
    batch_size: int = 4
):
    """
    Run VLM spatial prediction on onset frames.

    Full implementation reference:
    /home/disk3/Jiachen/accident/qwen35_spatial/run_full_test.py

    Args:
        onset_csv: CSV with predicted onset times
        video_dir: Directory containing videos
        output_csv: Output CSV path
        batch_size: VLM batch size
    """
    predictor = QwenSpatialPredictor()

    # Extract onset frames from videos
    # Run VLM prediction
    # Save results

    raise NotImplementedError("See accident/qwen35_spatial/run_full_test.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_spatial_inference(
        args.onset_csv,
        args.video_dir,
        args.output_csv,
        args.batch_size
    )
