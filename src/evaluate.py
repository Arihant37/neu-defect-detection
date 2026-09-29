"""Evaluate YOLOv8 model on NEU-DET validation set with detailed metrics."""
import argparse, json, time, os, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def evaluate(model_path, data_yaml, imgsz=640, conf=0.25, iou=0.45, save_dir=None):
    save_dir = Path(save_dir or ROOT / "models" / "evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    print(f"Model: {model_path} ({model.info()})")

    # Run validation
    start = time.time()
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    val_time = time.time() - start

    # Extract metrics
    box = metrics.box
    mp, mr, map50, map_ = box.mp, box.mr, box.map50, box.map
    per_class = []
    # Per-class metrics from confusion matrix
    cm_obj = metrics.confusion_matrix
    cm = cm_obj.matrix if hasattr(cm_obj, 'matrix') else (cm_obj() if callable(cm_obj) else None)
    for i, name in enumerate(CLASS_NAMES):
        if cm is not None and i < cm.shape[0]:
            tp = cm[i, i]
            fp = cm[i, :].sum() - tp
            fn = cm[i, -1] if cm.shape[1] > cm.shape[0] else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            per_class.append({"class": name, "precision": float(prec), "recall": float(rec),
                              "map50": 0, "map50_95": 0})
        else:
            per_class.append({"class": name, "precision": 0, "recall": 0, "map50": 0, "map50_95": 0})

    # Inference speed benchmark
    fps_results = {}
    for bs in [1]:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        times = []
        for _ in range(20):
            t0 = time.time()
            model(dummy, imgsz=imgsz, conf=conf, verbose=False)
            times.append((time.time() - t0) * 1000)
        fps_results[f"batch_{bs}"] = {"avg_ms": float(np.mean(times[5:])), "fps": float(1000 / np.mean(times[5:]))}

    results = {
        "model": str(model_path),
        "dataset": str(data_yaml),
        "mAP@0.5": float(map50),
        "mAP@0.5:0.95": float(map_),
        "precision": float(mp),
        "recall": float(mr),
        "per_class": per_class,
        "inference_speed": fps_results,
        "validation_time_s": float(val_time),
        "model_size_mb": float(os.path.getsize(model_path) / 1e6),
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Confusion matrix
    if cm is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        n = cm.shape[0]
        xlabels = CLASS_NAMES[:n] + ["background"] if n == len(CLASS_NAMES) else CLASS_NAMES[:n]
        ylabels = CLASS_NAMES[:n] + ["background"] if n == len(CLASS_NAMES) else CLASS_NAMES[:n]
        sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=xlabels, yticklabels=ylabels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix - NEU-DET Defect Detection")
        plt.tight_layout()
        fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    # Per-class bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    ax.bar(x - w/2, [c["precision"] for c in per_class], w, label="Precision")
    ax.bar(x + w/2, [c["recall"] for c in per_class], w, label="Recall")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_ylabel("Score"); ax.set_title("Per-Class Precision & Recall")
    ax.legend(); plt.tight_layout()
    fig.savefig(save_dir / "per_class_metrics.png", dpi=150)
    plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"mAP@0.5:       {map50:.4f}")
    print(f"mAP@0.5:0.95:   {map_:.4f}")
    print(f"Precision:      {mp:.4f}")
    print(f"Recall:         {mr:.4f}")
    print(f"Model size:     {results['model_size_mb']:.1f} MB")
    print(f"Inf speed:      {fps_results['batch_1']['fps']:.1f} FPS")
    print(f"\nPer-class:")
    for c in per_class:
        print(f"  {c['class']:18s} P={c['precision']:.3f} R={c['recall']:.3f} mAP50={c['map50']:.3f}")
    print(f"\nResults saved to: {save_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "neu_defect_run" / "weights" / "best.pt"))
    parser.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save-dir", default=str(ROOT / "models" / "evaluation"))
    args = parser.parse_args()
    evaluate(args.model, args.data, args.imgsz, args.conf, args.iou, args.save_dir)