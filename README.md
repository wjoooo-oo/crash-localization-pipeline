# Crash Localization Pipeline

**Spatiotemporal crash localization and classification using VideoMAE, Qwen VLM, and RT-DETR.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

This pipeline predicts three attributes of traffic crashes from video:
- **T (Temporal)**: When did the crash occur? (onset time in seconds)
- **S (Spatial)**: Where in the frame? (bounding box coordinates)
- **C (Classification)**: What type of crash? (head-on, rear-end, etc.)

**Leaderboard Performance (Submission 5):**
- Harmonic Mean: **0.53909**
- T: 0.6011 | S: 0.4828 | C: 0.6150 (200-GT eval)

## 🏗️ Architecture

```
┌─────────────┐
│ Input Video │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ [T] VideoMAE Temporal   │  ← ViT-Base, 16 frames
│  Predicts: onset_time   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ [S] Qwen VLM Spatial    │  ← 27B, 4-bit quantized
│  Predicts: bbox         │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ [C] RT-DETR + Rules     │  ← Detection + Reranker
│  Predicts: crash_type   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────┐
│ Submission  │
│    CSV      │
└─────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ VRAM (RTX 4090 / A6000 recommended)

### Quick Start

```bash
# Clone repository
git clone https://github.com/wjoooo-oo/crash-localization-pipeline.git
cd crash-localization-pipeline

# Install dependencies
pip install -r requirements.txt

# Download model checkpoints (see Setup below)
```

## 🚀 Usage

### 1. Download Checkpoints

**VideoMAE Temporal Model:**
```bash
# Download from GitHub Releases
wget https://github.com/wjoooo-oo/crash-localization-pipeline/releases/download/v1.0.0/videomae_sotad_best.tar.gz

# Extract to checkpoints directory
mkdir -p checkpoints
tar -xzf videomae_sotad_best.tar.gz -C checkpoints/

# Verify extraction
ls checkpoints/checkpoints_sotad_temporal/best_model/
# Should see: model.safetensors, config.json, preprocessor_config.json
```

**Qwen VLM & RT-DETR:** Auto-downloaded on first run

### 2. Run Full Pipeline

```bash
python scripts/run_pipeline.py \
  --video-dir /path/to/test/videos \
  --temporal-checkpoint checkpoints/videomae_sotad/best_model \
  --output-dir outputs
```

**Output:** `outputs/submission.csv`

### 3. Evaluate on 200-GT Subset

```bash
python scripts/evaluate.py \
  --pred-csv outputs/submission.csv \
  --gt-csv data/ground_truth_200.csv
```

**Expected Output:**
```
T: 0.6011
S: 0.4828
C: 0.6150
Harmonic Mean: 0.5391
```

## 📂 Project Structure

```
crash-localization-pipeline/
├── src/
│   ├── temporal/
│   │   ├── train.py            # VideoMAE training
│   │   └── inference.py        # Temporal prediction
│   ├── spatial/
│   │   └── vlm_predictor.py    # Qwen VLM spatial localization
│   └── classification/
│       └── type_classifier.py  # RT-DETR + rules-based classifier
│
├── scripts/
│   ├── run_pipeline.py         # Full T→S→C pipeline
│   └── evaluate.py             # Evaluation on 200-GT
│
├── configs/
│   └── default.yaml            # Default configuration
│
├── data/
│   ├── sotad_train_temporal.csv
│   ├── sotad_val_temporal.csv
│   └── ground_truth_200.csv
│
├── docs/
│   ├── ARCHITECTURE.md         # Detailed architecture docs
│   ├── TRAINING.md             # Training guide
│   └── API.md                  # API reference
│
├── requirements.txt
└── README.md
```

## 🔧 Advanced Usage

### Training Temporal Model

```bash
python src/temporal/train.py \
  --train-csv data/sotad_train_temporal.csv \
  --val-csv data/sotad_val_temporal.csv \
  --output-dir checkpoints/my_model \
  --batch-size 32 \
  --epochs 40
```

### Custom VLM Prompt

Edit `src/spatial/vlm_predictor.py`:

```python
PROMPT_TEMPLATE = """
Your custom prompt here...
Return JSON: {"bbox": [x_min, y_min, x_max, y_max], "confidence": float}
"""
```

### Individual Stage Inference

```bash
# Stage 1: Temporal only
python src/temporal/inference.py \
  --checkpoint checkpoints/videomae_sotad/best_model \
  --video-dir /path/to/videos \
  --output-csv temporal_pred.csv

# Stage 2: Spatial only
python src/spatial/vlm_predictor.py \
  --onset-csv temporal_pred.csv \
  --video-dir /path/to/videos \
  --output-csv spatial_pred.csv

# Stage 3: Classification only
python src/classification/type_classifier.py \
  --temporal-csv temporal_pred.csv \
  --spatial-csv spatial_pred.csv \
  --video-dir /path/to/videos \
  --output-csv type_pred.csv
```

## 📊 Performance

### Hardware Requirements

| Component | Training | Inference |
|-----------|----------|-----------|
| GPU Memory | 16GB+ | 24GB+ |
| RAM | 32GB+ | 32GB+ |
| Storage | 500GB+ | 100GB+ |

### Speed Benchmarks (RTX 4090, 2000 videos)

| Stage | Time | GPU Mem |
|-------|------|---------|
| Temporal | 30 min | 4GB |
| Spatial (VLM) | 2 hours | 18GB |
| Classification | 1 hour | 2GB |
| **Total** | **3.5 hours** | **18GB peak** |

### Accuracy (200-GT Subset)

| Metric | Sub5 | Target |
|--------|------|--------|
| T (Temporal) | 0.6011 | 0.65+ |
| S (Spatial) | 0.4828 | 0.55+ |
| C (Classification) | 0.6150 | 0.65+ |
| **Harmonic Mean** | **0.5391** | **0.60+** |

## 📚 Documentation

- [Architecture Details](docs/ARCHITECTURE.md) - Deep dive into model designs
- [Training Guide](docs/TRAINING.md) - How to train from scratch
- [API Reference](docs/API.md) - Code API documentation

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **VideoMAE**: [MCG-NJU/videomae](https://github.com/MCG-NJU/VideoMAE)
- **Qwen VLM**: [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- **RT-DETR**: [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📧 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/wjoooo-oo/crash-localization-pipeline/issues).

## 📖 Citation

```bibtex
@misc{crash-localization-2026,
  title={Spatiotemporal Crash Localization and Classification},
  author={wenjie},
  year={2026},
  url={https://github.com/wjoooo-oo/crash-localization-pipeline}
}
```

---

**Note:** This is a research implementation. Model checkpoints trained on proprietary datasets cannot be publicly released. The code framework and methodology are provided for reference.

## 🔗 Related Work

- [Original Implementation](https://github.com/wjoooo-oo/original-repo) - Full research codebase
- [Competition Page](https://kaggle.com/competitions/...) - Dataset and leaderboard

## 🐛 Known Issues

1. **Long videos (60+ sec):** Temporal accuracy degrades - see [Issue #12](https://github.com/wjoooo-oo/crash-localization-pipeline/issues/12)
2. **Occluded crashes:** VLM struggles with smoke/debris - [Issue #7](https://github.com/wjoooo-oo/crash-localization-pipeline/issues/7)
3. **Rare crash types:** Classification biased toward common types - [Issue #5](https://github.com/wjoooo-oo/crash-localization-pipeline/issues/5)

## 🗺️ Roadmap

- [ ] Multi-frame VLM ensembling for spatial
- [ ] Learned type classifier (replace rules)
- [ ] Real-time inference optimization
- [ ] Web demo deployment
