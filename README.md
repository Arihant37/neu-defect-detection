# NEU-DET Steel Surface Defect Detection

**End-to-end computer vision pipeline for industrial quality control** — from dataset curation to production deployment. Built with YOLOv8 and designed to showcase skills relevant to [Roboflow](https://roboflow.com).

## Pipeline Overview

```
Pascal VOC XML  ──►  YOLO Conversion  ──►  Class Imbalance Handling  ──►  YOLOv8 Training
       │                                                                 │
       ▼                                                                 ▼
NEU-DET Dataset                                              mAP@0.5: 0.695
                                                             Precision: 0.603
                                                             Recall: 0.688
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                         ▼
                   FastAPI Inference Server              Gradio Web UI
                   (:8000)                              (:7860)
                          │                             │
                          └─────────────────────────────┘
                                          │
                                          ▼
                                  Docker Container
                            (port 8000 + 7860)
```

## What It Does

Detects **6 types of steel surface defects** in real-time:
- **Crazing** — micro-cracks in surface coating
- **Inclusion** — foreign material embedded in steel
- **Patches** — surface patch irregularities
- **Pitted Surface** — corrosion pits
- **Rolled-in Scale** — oxide scale from rolling process
- **Scratches** — mechanical surface damage

## Key Technical Decisions

### Class Imbalance Handling
The NEU-DET dataset has significant imbalance (inclusion: 852 instances vs pitted_surface: 345). I implemented **two complementary strategies**:
1. **Weighted Random Sampling** — oversamples images containing rare classes using per-image weights based on geometric mean of inverse class frequencies
2. **Per-class Focal Loss** — custom loss subclass that applies per-class alpha and gamma to the BCE classification term, mitigating the dominance of frequent classes

### Architecture Choice
- **YOLOv8** (Ultralytics) — production-ready, well-supported, exports to ONNX
- **Why not RF-DETR?** YOLOv8's ecosystem maturity and TensorBoard/ONNX export make it more practical for a demonstration project; RF-DETR would be the next step for production

### Deployment Strategy
- **FastAPI** for low-latency inference server with CORS
- **Gradio** for interactive web UI with webcam support
- **Docker** for containerized deployment
- **ONNX export** for edge deployment compatibility

## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.695** |
| mAP@0.5:0.95 | **0.368** |
| Precision | **0.603** |
| Recall | **0.688** |
| Model Size | 5.4 MB (PT) / 9.8 MB (ONNX) |

### Per-Class Performance
| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| crazing | - | - | - |
| inclusion | - | - | - |
| patches | - | - | - |
| pitted_surface | - | - | - |
| rolled-in_scale | - | - | - |
| scratches | - | - | - |

*(Detailed per-class metrics in `models/evaluation/metrics.json`)*

## Why This Matters to Roboflow

1. **End-to-end pipeline** — demonstrates ownership from data to deployment, matching Roboflow's "full-stack people" value
2. **Class imbalance awareness** — directly addresses a core challenge in industrial CV that Roboflow tackles
3. **Production-ready code** — FastAPI + Docker + ONNX export shows capability to ship real solutions
4. **Neu-DET dataset** — industrial quality control application aligns with Roboflow's manufacturing focus
5. **Comparison-aware** — built with awareness of RF-DETR as the state-of-the-art alternative

## Installation

```bash
# Clone and setup
git clone <repo_url>
cd neu-defect-detection
pip install -r requirements.txt

# Run evaluation (optional - model already trained)
python3 src/evaluate.py --model models/neu_defect_run/weights/best.pt

# Start inference server
python3 src/deploy.py --host 0.0.0.0 --port 8000

# Start web UI
python3 app/app.py
```

## Docker

```bash
docker build -t neu-defect-detection .
docker run -p 8000:8000 -p 7860:7860 neu-defect-detection
```

## API Endpoints

- `GET /health` — service health check
- `GET /metrics` — model performance metrics
- `POST /detect` — upload image, get detections (JSON with annotated image)

## Skills Demonstrated

- **Computer Vision**: YOLOv8, object detection, class imbalance handling
- **Data Pipeline**: XML→YOLO conversion, dataset validation
- **MLOps**: TensorBoard logging, model versioning, ONNX export
- **Deployment**: FastAPI, Docker, REST API design
- **Evaluation**: mAP metrics, confusion matrix, per-class analysis

## License

Dataset: CC BY 4.0 (NEU Surface Defect Database)
Code: MIT