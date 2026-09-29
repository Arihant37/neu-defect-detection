#!/usr/bin/env python3
"""
Steel Surface Defect Detection — YOLOv8 Training Script
========================================================
Trains an ultralytics YOLOv8 model on the NEU-DET steel surface defect dataset.

Handles class imbalance via:
  - Weighted random sampling of images (based on dominant/unique class frequencies)
  - Per-class focal loss coefficients (configurable via --class-weights)

Outputs:
  - Best and last checkpoints to models/<run>/weights/
  - TensorBoard logs to models/<run>/logs/
  - CSV results to models/<run>/results.csv
  - ONNX export to models/<run>/weights/exported/

Usage:
    python3 src/train.py --epochs 50 --img-size 640 --batch 16
    python3 src/train.py --epochs 100 --data ../data.yaml --project models --name steel_v8
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.loss import v8DetectionLoss, make_anchors

# ---------------------------------------------------------------------------
# Section 1: Weighted-Random Sampler for Class Imbalance
# ---------------------------------------------------------------------------

def compute_class_weights(labels_dir: Path, verbose: bool = True) -> tuple[list[float], dict[int, int]]:
    """Compute inverse-frequency weights for each image based on its classes.

    Parameters
    ----------
    labels_dir : Path
        Directory containing YOLO-format label .txt files.
    verbose : bool
        If True, print the computed class distribution.

    Returns
    -------
    sample_weights : list[float]
        Per-image weight used by WeightedRandomSampler.
    class_counts : dict[int, int]
        Total instance count per class (used for logging).
    """
    counts: Counter = Counter()
    image_classes: list[list[int]] = []  # one entry per image file

    for label_path in sorted(labels_dir.glob("*.txt")):
        classes: list[int] = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                counts[cls] += 1
                classes.append(cls)
        image_classes.append(classes)

    n_images = len(image_classes)
    n_classes = max(counts) + 1 if counts else 0

    # Total instances per class
    total_instances = sum(counts.values())
    inv_freq = {c: total_instances / max(counts.get(c, 1), 1) for c in range(n_classes)}

    # Per-image weight: use the harmonic mean of the inverse frequencies of
    # all classes present in that image so images with multiple minority classes
    # are also sampled more often.
    sample_weights: list[float] = []
    for classes in image_classes:
        unique_cls = set(classes)
        # Geometric mean of inverse frequencies
        log_w = sum(math.log(inv_freq.get(c, 1.0)) for c in unique_cls) / len(unique_cls)
        sample_weights.append(math.exp(log_w))

    # Normalize so mean weight ≈ 1 (preserves effective batch size)
    mean_w = sum(sample_weights) / len(sample_weights) if sample_weights else 1.0
    sample_weights = [w / mean_w for w in sample_weights]

    if verbose:
        LOGGER.info(f"Class distribution ({n_classes} classes, {n_images} images):")
        names: list[str] = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
        for c in range(n_classes):
            LOGGER.info(f"  {names[c] if c < len(names) else f'class_{c}':18s}: {counts[c]} instances  weight={inv_freq[c]:.3f}")
        LOGGER.info(f"Total instances: {total_instances}, mean sample weight: 1.000")

    return sample_weights, dict(counts)


# ---------------------------------------------------------------------------
# Section 2: Dataset-aware Weighted DataLoader wrapper
# ---------------------------------------------------------------------------

def build_weighted_dataloader(
    cfg,
    img_path: str,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler,
    mode: str = "train",
    data: dict | None = None,
) -> DataLoader:
    """Build a DataLoader that uses a WeightedRandomSampler instead of uniform shuffle.

    This is used during training so rare-class images appear more frequently.
    Validation uses a standard sequential DataLoader (no sampling bias).
    """
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    import os as _os

    dataset = build_yolo_dataset(cfg, img_path, batch_size, data or {}, mode=mode)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)

    if mode == "train":
        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
        )
    return loader


# ---------------------------------------------------------------------------
# Section 3: Custom Trainer with Class-Imbalance Support
# ---------------------------------------------------------------------------

class FocalDetectionLoss(v8DetectionLoss):
    """YOLOv8 detection loss with per-class focal weighting.

    Extends v8DetectionLoss by applying a per-class alpha and gamma to the
    BCE classification loss.  This is the standard focal-loss formulation:
        p_t = sigmoid(x_t) if y=1 else 1 - sigmoid(x_t)
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None,
                 alpha: list[float] | None = None, gamma: float = 2.0):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.alpha = (torch.tensor(alpha, device=self.device, dtype=torch.float32)
                      if alpha is not None else None)
        self.gamma = gamma

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Compute per-class focal loss on the classification term."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Focal BCE classification loss (per-class alpha and gamma)
        # Standard focal: alpha_t * (1 - p_t)^gamma * log(p_t), with p_t = target_scores
        one_minus = 1.0 - target_scores
        focal_weight = one_minus.pow(self.gamma)
        if self.alpha is not None:
            # broadcast per-class alpha over the class dimension
            alpha_t = self.alpha.view(1, 1, -1).expand_as(target_scores)
            focal_weight = focal_weight * alpha_t
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)) * focal_weight).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class SteelDefectTrainer(DetectionTrainer):
    """YOLOv8 trainer with class-imbalance handling for NEU-DET.

    Applies two complementary strategies:
      1. Weighted random sampling — images containing rare-defect classes are
         oversampled during training so the model sees them more often.
      2. Per-class focal loss — each class gets its own BCE alpha and gamma
         so minority classes contribute proportionally more to the loss.
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None):
        from ultralytics.utils import DEFAULT_CFG
        self._sample_weights: list[float] | None = None
        self._class_counts: dict[int, int] | None = None
        self._loss_weights: dict[str, list[float]] | None = None
        # Ensure `model` is in overrides so the base trainer resolves it.
        if overrides and "model" not in overrides and overrides.get("pretrained"):
            overrides = {**overrides, "model": overrides["pretrained"]}
        super().__init__(cfg or DEFAULT_CFG, overrides, _callbacks)

    # ------------------------------------------------------------------
    # Override: resolve dataset paths relative to the project root.
    # The base trainer resolves paths relative to the YAML file's parent,
    # but the YAML uses relative paths that only work when cwd == project root.
    # ------------------------------------------------------------------
    def get_dataset(self):
        """Build the dataset dict with absolute paths resolved from the project root."""
        data = super().get_dataset()
        project_root = Path(__file__).resolve().parent.parent
        for key in ("train", "val", "test"):
            if key in data and data[key]:
                p = Path(data[key])
                if not p.is_absolute():
                    data[key] = str((project_root / p).resolve())
        return data

    # ------------------------------------------------------------------
    # Override 1: Build a dataloader with weighted sampling for training
    # ------------------------------------------------------------------
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode == "train" and self._sample_weights is not None and rank == -1:
            num_samples = len(self._sample_weights)
            sampler = WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=num_samples,
                replacement=True,
            )
            return build_weighted_dataloader(
                self.args, dataset_path, batch_size,
                num_workers=self.args.workers,
                sampler=sampler,
                mode="train",
                data=self.data,
            )
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    # ------------------------------------------------------------------
    # Override 2: Save metrics safely (handles DetMetrics object)
    # ------------------------------------------------------------------
    def save_metrics(self, metrics):
        """Save training metrics to CSV, handling both dict and DetMetrics objects."""
        if hasattr(metrics, "results_dict"):
            metrics = metrics.results_dict
        if hasattr(metrics, "keys"):
            keys, vals = list(metrics.keys()), list(metrics.values())
        else:
            keys, vals = ["loss"], [float(metrics)]
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    # ------------------------------------------------------------------
    # Helper: Export ONNX model after training
    # ------------------------------------------------------------------
    def export_model(self):
        """Export the best model to ONNX format for edge deployment."""
        export_dir = self.save_dir / "weights" / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"\nExporting best model to ONNX: {export_dir}")
        try:
            from ultralytics import YOLO as YOLOWrapper
            best_ckpt = self.best
            yolo_export = YOLOWrapper(str(best_ckpt))
            yolo_export.export(
                format="onnx",
                imgsz=self.args.imgsz,
                half=False,            # FP32 for maximum edge compatibility
                dynamic=False,          # Fixed size ONNX for edge devices
                simplify=True,          # Apply ONNX-Simplifier if available
                opset=12,
            )
            # Ultralytics writes the ONNX file to the weights dir, not the
            # export sub-dir, so check both locations.
            exported_files = list(export_dir.glob("*.onnx"))
            if not exported_files:
                exported_files = list((export_dir.parent).glob("*.onnx"))
            if exported_files:
                for f in exported_files:
                    LOGGER.info(f"ONNX export saved to {f} ({f.stat().st_size / 1e6:.1f} MB)")
            else:
                LOGGER.warning("ONNX export did not produce output file.")
        except Exception as e:
            LOGGER.error(f"ONNX export failed: {e}")
            LOGGER.info("Model weights are still available at best.pt and last.pt")


# ---------------------------------------------------------------------------
# Section 4: TensorBoard + Progress Logging Callbacks
# ---------------------------------------------------------------------------

def setup_tensorboard_logs(save_dir: Path, run_name: str) -> None:
    """Attach TensorBoard and CSV logging callbacks to a YOLO trainer instance.

    Parameters
    ----------
    save_dir : Path
        Directory returned by the trainer (e.g. runs/detect/exp1).
    run_name : str
        Name passed via --name; used to build the TensorBoard log path.
    """
    from torch.utils.tensorboard import SummaryWriter

    tb_dir = Path(save_dir) / "logs" / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    def _log_scalar(tag: str, value: float, epoch: int) -> None:
        writer.add_scalar(tag, value, epoch)

    def on_fit_epoch_end(trainer) -> None:
        epoch = trainer.epoch
        metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
        lr = trainer.lr if hasattr(trainer, "lr") else {}

        # Loss components
        try:
            metrics = metrics.results_dict if hasattr(metrics, "results_dict") else metrics
        except Exception:
            pass
        if hasattr(metrics, "items"):
            for key, val in metrics.items():
                try:
                    _log_scalar(f"loss/{key}", float(val), epoch)
                except (TypeError, ValueError):
                    pass
        for key, val in lr.items():
            _log_scalar(f"lr/{key}", float(val), epoch)

        # Validation metrics (written by ultralytics trainer)
        val_metrics = trainer.validator.metrics if hasattr(trainer, "validator") else {}
        try:
            val_metrics = val_metrics.results_dict if hasattr(val_metrics, "results_dict") else val_metrics
        except Exception:
            pass
        if hasattr(val_metrics, "items"):
            for key, val in val_metrics.items():
                if isinstance(val, (int, float)):
                    _log_scalar(f"val/{key}", float(val), epoch)

    def on_train_epoch_end(trainer) -> None:
        # Log training loss running average
        if trainer.tloss is not None:
            try:
                train_loss = float(torch.mean(trainer.tloss))
                _log_scalar("loss/train_total", train_loss, trainer.epoch)
            except (TypeError, ValueError):
                pass

    def register_callbacks(trainer):
        trainer.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        trainer.add_callback("on_train_epoch_end", on_train_epoch_end)

    return register_callbacks, writer


def print_detailed_metrics(trainer) -> None:
    """Print per-class and aggregate metrics to stdout after training."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("TRAINING COMPLETE — DETAILED METRICS")
    LOGGER.info("=" * 80)

    # Epoch metrics
    metrics = trainer.metrics
    if metrics:
        LOGGER.info(f"\nFinal epoch metrics:")
        LOGGER.info(f"  mAP@0.5   : {metrics.get('metrics/mAP50(B)', 'N/A')}")
        LOGGER.info(f"  mAP@0.5:0.95 : {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        LOGGER.info(f"  Precision : {metrics.get('metrics/precision(B)', 'N/A')}")
        LOGGER.info(f"  Recall    : {metrics.get('metrics/recall(B)', 'N/A')}")

    # Per-class metrics from validator (if available)
    if hasattr(trainer, "validator") and trainer.validator:
        vm = trainer.validator.metrics
        if hasattr(vm, "box"):
            box = vm.box
            names = list(trainer.data.get("names", {}).values())
            LOGGER.info(f"\nPer-class metrics (final validation):")
            header = f"{'Class':<18} {'Prec':>7} {'Recall':>7} {'mAP@50':>7} {'mAP50-95':>8}"
            LOGGER.info(header)
            LOGGER.info("-" * len(header))
            for i in range(len(names)):
                try:
                    p, r, ap50, ap = box.class_result(i)
                    LOGGER.info(f"{names[i]:<18} {p:>7.3f} {r:>7.3f} {ap50:>7.3f} {ap:>8.3f}")
                except Exception:
                    LOGGER.info(f"{names[i]:<18} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>8}")

    # Model info
    best = trainer.best
    if best.exists():
        LOGGER.info(f"\nBest model : {best}")
        # Count parameters from the trained model (EMA state dict)
        try:
            param_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            LOGGER.info(f"  Parameters: {param_count:,}")
        except Exception:
            pass
        LOGGER.info(f"  File size : {best.stat().st_size / 1e6:.1f} MB")
    else:
        LOGGER.warning("Best model checkpoint not found.")

    LOGGER.info("=" * 80)


# ---------------------------------------------------------------------------
# Section 5: Hardware & Runtime Info
# ---------------------------------------------------------------------------

def print_runtime_info(args: argparse.Namespace, project_dir: Path, run_name: str) -> None:
    """Print GPU and runtime environment details before training starts."""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("RUNTIME INFORMATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"PyTorch version : {torch.__version__}")
    LOGGER.info(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU             : {torch.cuda.get_device_name(0)}")
        LOGGER.info(f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        LOGGER.info(f"CUDA version    : {torch.version.cuda}")
    else:
        LOGGER.info("Running on CPU (training will be slower).")
    LOGGER.info(f"Dataset images  : train={args.train_images}, val={args.val_images}")
    LOGGER.info(f"Output dir      : {project_dir / run_name}")
    LOGGER.info("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Section 6: CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on NEU-DET steel surface defect dataset with class-imbalance handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset paths
    parser.add_argument("--data-yaml", type=str, default="data.yaml",
                        help="Path to data.yaml configuration file")
    parser.add_argument("--train-images", type=int, default=1439,
                        help="Expected number of training images (for logging)")
    parser.add_argument("--val-images", type=int, default=360,
                        help="Expected number of validation images (for logging)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--img-size", "--imgsz", type=int, default=640,
                        help="Input image size (square)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of DataLoader worker threads")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device(s), e.g. '0', '0,1', or 'cpu'")
    parser.add_argument("--optimizer", type=str, default="SGD",
                        choices=["SGD", "Adam", "AdamW"],
                        help="Optimizer for training")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (SGD)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.937,
                        help="Momentum for SGD")
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
                        help="Number of warmup epochs (fraction of total if < 1)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Epochs to wait for no improvement before early stopping")

    # Augmentation hyperparameters
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Mosaic augmentation probability (0-1)")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="MixUp augmentation probability (0-1)")
    parser.add_argument("--copy-paste", type=float, default=0.0,
                        help="Copy-Paste augmentation probability (0-1)")
    parser.add_argument("--degrees", type=float, default=10.0,
                        help="Random rotation degrees")
    parser.add_argument("--translate", type=float, default=0.1,
                        help="Random translation fraction")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Random scaling factor")
    parser.add_argument("--shear", type=float, default=0.0,
                        help="Random shear angle")
    parser.add_argument("--flipud", type=float, default=0.0,
                        help="Vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="Horizontal flip probability")
    parser.add_argument("--hsv-h", type=float, default=0.015,
                        help="HSV hue jitter")
    parser.add_argument("--hsv-s", type=float, default=0.7,
                        help="HSV saturation jitter")
    parser.add_argument("--hsv-v", type=float, default=0.4,
                        help="HSV value jitter")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="Disable mosaic for the last N epochs before training ends")

    # Class imbalance
    parser.add_argument("--class-weights", type=float, nargs="+", default=None,
                        help="Manual per-class weights for focal loss (6 floats). "
                             "Overrides auto-computed weights.")
    parser.add_argument("--use-weighted-sampling", type=lambda x: x.lower() in ("1", "true", "yes"),
                        default=True, help="Use WeightedRandomSampler for training")
    parser.add_argument("--focal-alpha", type=float, default=None,
                        help="Focal loss alpha per class. Overrides auto-computed.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (generalization parameter)")

    # Output / logging
    parser.add_argument("--project", type=str, default="models",
                        help="Project directory for saving models and logs")
    parser.add_argument("--name", type=str, default="steel_v8",
                        help="Run name (sub-directory under project)")
    parser.add_argument("--exist-ok", action="store_true",
                        help="Overwrite existing run directory")
    parser.add_argument("--save-period", type=int, default=-1,
                        help="Save checkpoint every N epochs (-1 = only best/last)")
    parser.add_argument("--log-csv", type=lambda x: x.lower() in ("1", "true", "yes"),
                        default=True, help="Save results.csv alongside TensorBoard logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Model selection
    parser.add_argument("--pretrained", type=str, default="yolo26n.pt",
                        help="Pretrained weight filename or path (default: yolo26n.pt from torch hub)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Section 7: Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve absolute paths
    project_dir = Path(__file__).resolve().parent.parent / args.project
    run_dir = project_dir / args.name
    data_yaml = Path(__file__).resolve().parent.parent / "data" / args.data_yaml
    labels_dir = Path(__file__).resolve().parent.parent / "data" / "train" / "labels"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found at {labels_dir}")

    # Print runtime and hardware info
    print_runtime_info(args, project_dir, args.name)

    # ---- Compute class weights for imbalance handling ----
    LOGGER.info("Computing class distribution and sample weights...")
    sample_weights, class_counts = compute_class_weights(labels_dir, verbose=True)

    # Compute loss weights (alpha per class) from inverse class frequency
    n_total = sum(class_counts.values())
    n_classes = len(class_counts)
    if args.class_weights is not None:
        alpha_per_class = args.class_weights[:n_classes]
        LOGGER.info(f"Using provided class weights: {alpha_per_class}")
    else:
        alpha_per_class = [n_total / max(class_counts.get(c, 1), 1) for c in range(n_classes)]
        # Normalize alpha to mean=1 so loss scale is preserved
        alpha_per_class = [w / (sum(alpha_per_class) / len(alpha_per_class)) for w in alpha_per_class]
    LOGGER.info(f"Loss alpha per class: {[f'{w:.3f}' for w in alpha_per_class]}")
    LOGGER.info(f"Focal gamma: {args.focal_gamma}")

    # ---- Configure trainer overrides ----
    overrides = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.img_size,
        "device": args.device,
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "save_period": args.save_period,
        "seed": args.seed,
        "deterministic": True,
        "close_mosaic": args.close_mosaic,
        "pretrained": args.pretrained,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "warmup_epochs": args.warmup_epochs,
        # Augmentation
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        # Disable automatic pip update check to avoid network issues
        "verbose": True,
    }

    # ---- Initialize trainer and inject class-imbalance parameters ----
    trainer = SteelDefectTrainer(overrides=overrides)
    trainer._sample_weights = sample_weights
    trainer._class_counts = class_counts
    # ---- Set up TensorBoard / progress callbacks ----
    tb_setup, tb_writer = setup_tensorboard_logs(run_dir, args.name)
    tb_setup(trainer)

    # Override init_criterion to use FocalDetectionLoss with per-class alpha/gamma
    def patched_init_criterion():
        criterion = FocalDetectionLoss(
            trainer.model,
            alpha=alpha_per_class,
            gamma=args.focal_gamma,
        )
        return criterion

    trainer.init_criterion = patched_init_criterion

    # Register post-training ONNX export callback
    def _on_train_end(trainer_):
        trainer.export_model()
    trainer.add_callback("on_train_end", _on_train_end)

    # ---- Start training ----
    LOGGER.info(f"Starting YOLOv8 training on {data_yaml}")
    LOGGER.info(f"  Epochs    : {args.epochs}")
    LOGGER.info(f"  Batch     : {args.batch}")
    LOGGER.info(f"  Image size: {args.img_size}")
    LOGGER.info(f"  Device    : {args.device}")
    LOGGER.info(f"  Optimizer : {args.optimizer}")
    LOGGER.info(f"  LR        : {args.lr0}")
    LOGGER.info(f"  Weight decay: {args.weight_decay}")
    LOGGER.info(f"  Mosaic    : {args.mosaic}")
    LOGGER.info(f"  Fliplr    : {args.fliplr}")
    LOGGER.info("")

    try:
        trainer.train()
    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
        raise

    # ---- Finalize logging ----
    if tb_writer is not None:
        tb_writer.close()
        LOGGER.info("TensorBoard logging saved.")

    # Print detailed metrics
    print_detailed_metrics(trainer)

    # Summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Best model    : {trainer.best}")
    LOGGER.info(f"Last model    : {trainer.last}")
    LOGGER.info(f"Class counts  : {dict(class_counts)}")
    LOGGER.info(f"Sample weights: {len(sample_weights)} images, range=[{min(sample_weights):.3f}, {max(sample_weights):.3f}]")
    LOGGER.info(f"Alpha (loss)  : {[f'{w:.3f}' for w in alpha_per_class]}")
    LOGGER.info(f"Logs dir      : {run_dir}")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
