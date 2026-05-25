import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["DISPLAY"] = ""

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "ShelfSight API is running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image)

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