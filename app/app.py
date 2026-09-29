"""
Gradio web application for NEU-DET Steel Surface Defect Detection.
Upload an image or use the webcam to detect steel surface defects in real-time.
"""
import os
import sys
import time
from pathlib import Path

import gradio as gr
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pt"

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_COLORS = {
    "crazing": (0, 255, 255),       # Yellow
    "inclusion": (255, 0, 255),     # Magenta
    "patches": (255, 255, 0),       # Cyan
    "pitted_surface": (255, 0, 0),  # Red
    "rolled-in_scale": (0, 165, 255), # Orange
    "scratches": (0, 255, 0),       # Green
}

_model = None


def load_model():
    """Lazy-load the YOLOv8 model."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first: python3 src/train.py")
        _model = YOLO(str(MODEL_PATH))
    return _model


def detect_image(image, conf_threshold=0.25, iou_threshold=0.45):
    """Run detection on an uploaded image."""
    if image is None:
        return None, None, "No image provided."

    model = load_model()

    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    start = time.time()
    results = model(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
    inference_time = (time.time() - start) * 1000

    result = results[0]
    annotated = result.plot()

    # Build detection summary
    boxes = result.boxes
    if len(boxes) == 0:
        summary = "No defects detected."
    else:
        lines = [f"**Detected {len(boxes)} defect(s)** in {inference_time:.1f} ms\n"]
        cls_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
        for cls_name, count in sorted(cls_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cls_name}**: {count}")
        summary = "\n".join(lines)

    return annotated, None, summary


def detect_webcam(image, conf_threshold=0.25, iou_threshold=0.45):
    """Run detection on webcam feed."""
    return detect_image(image, conf_threshold, iou_threshold)


def compare_with_ground_truth(image, label_path, conf_threshold=0.25):
    """Compare detection results with ground truth annotations."""
    if image is None:
        return None, "No image provided."

    model = load_model()

    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    results = model(img_array, conf=conf_threshold, verbose=False)
    annotated = results[0].plot()

    gt_text = ""
    if label_path and os.path.exists(label_path):
        with open(label_path) as f:
            lines = f.readlines()
        gt_text = f"**Ground Truth:** {len(lines)} defect(s)\n"
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                cls_id = int(parts[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                gt_text += f"- {cls_name}\n"

    return annotated, gt_text


# Build the Gradio interface
with gr_blocks := gr.Blocks(title="NEU-DET Steel Defect Detection") as demo:
    gr.Markdown("# 🔍 NEU-DET Steel Surface Defect Detection")
    gr.Markdown("Detect 6 types of steel surface defects using YOLOv8. "
                "Upload an image or use the webcam to identify manufacturing defects.")

    with gr.Tab("Image Detection"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Steel Surface Image")
            with gr.Column():
                conf_slider = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="Confidence Threshold")
                iou_slider = gr.Slider(0.01, 1.0, value=0.45, step=0.01, label="IoU Threshold")
                detect_btn = gr.Button("Detect Defects", variant="primary")
        output_image = gr.Image(label="Detection Result")
        output_text = gr.Markdown(label="Results")

    with gr.Tab("Webcam"):
        webcam_input = gr.Image(source="webcam", type="numpy", label="Webcam Feed")
        webcam_btn = gr.Button("Detect from Webcam", variant="primary")
        webcam_output = gr.Image(label="Webcam Result")
        webcam_text = gr.Markdown(label="Webcam Results")

    with gr.Tab("Compare with Ground Truth"):
        compare_image = gr.Image(type="numpy", label="Image with Ground Truth Labels")
        label_input = gr.Textbox(label="Ground Truth Label File Path (optional)")
        compare_btn = gr.Button("Compare", variant="secondary")
        compare_output = gr.Image(label="Comparison Result")
        compare_text = gr.Markdown(label="Comparison")

    detect_btn.click(detect_image, [image_input, conf_slider, iou_slider], [output_image, gr.State(), output_text])
    webcam_btn.click(detect_webcam, [webcam_input, conf_slider, iou_slider], [webcam_output, gr.State(), webcam_text])
    compare_btn.click(compare_with_ground_truth, [compare_image, label_input, conf_slider], [compare_output, compare_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)