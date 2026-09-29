"""FastAPI deployment for NEU-DET steel surface defect detection."""
import argparse, io, json, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

app = FastAPI(title="NEU-DET Steel Defect Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model: Optional[YOLO] = None
_model_path: str = None
_inference_times: list = []


def load_model(model_path: str):
    global _model, _model_path
    if _model is None or _model_path != model_path:
        _model = YOLO(model_path)
        _model_path = model_path
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _model(dummy, imgsz=640, verbose=False)
    return _model


@app.on_event("startup")
async def startup():
    model_path = os.environ.get("MODEL_PATH", str(ROOT / "models" / "neu_defect_run" / "weights" / "best.pt"))
    load_model(model_path)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": _model_path, "classes": CLASS_NAMES}


@app.get("/metrics")
async def metrics():
    avg_ms = float(np.mean(_inference_times)) if _inference_times else 0
    return {
        "model": _model_path,
        "total_requests": len(_inference_times),
        "avg_inference_ms": round(avg_ms, 2),
        "fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    model = load_model(_model_path)
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_array = np.array(image)
    t0 = time.time()
    results = model(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
    inference_ms = (time.time() - t0) * 1000
    _inference_times.append(inference_ms)

    result = results[0]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
            "class_id": cls_id,
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 1) for v in xyxy],
        })

    annotated = result.plot()
    _, buf = cv2.imencode(".jpg", annotated)
    import base64
    annotated_b64 = base64.b64encode(buf).decode()

    return JSONResponse({
        "detections": detections,
        "num_detections": len(detections),
        "inference_time_ms": round(inference_ms, 2),
        "image_size": image.size,
        "annotated_image_b64": f"data:image/jpeg;base64,{annotated_b64}",
    })


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=str(ROOT / "models" / "neu_defect_run" / "weights" / "best.pt"))
    args = parser.parse_args()
    os.environ["MODEL_PATH"] = args.model
    uvicorn.run(app, host=args.host, port=args.port)