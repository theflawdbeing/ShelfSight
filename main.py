import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "ShelfSight API is running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)
    results = model(image_array)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": results[0].names[int(box.cls)],
            "confidence": round(float(box.conf), 2),
            "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
        })

    return {
        "total_detections": len(detections),
        "detections": detections
    }