from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from uuid import uuid4
import logging
import base64
from detection import detect_resistors, calculate_global_resistance

app = FastAPI(
    title="EduVision API",
    description="API for detecting resistors and calculating global resistance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResistorData(BaseModel):
    id: int
    colors: List[str]
    resistance: float
    tolerance: float
    orientation: str
    bbox: List[float]


class DetectionResponse(BaseModel):
    resistors: List[ResistorData]
    global_resistance: float
    image_id: str

@app.on_event("startup")
async def startup_event():
    app.state.model = YOLO("../validation/best.pt")
    app.state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app.state.model.to(app.state.device)
    logging.info(f"Model loaded on device: {app.state.device}")


@app.post("/detect", response_model=DetectionResponse)
async def detect(
        data: dict = Body(...)
):

    try:
        base64_image = data.get("frame")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No frame data provided")

        encoded_data = base64_image.split(',')[1] if ',' in base64_image else base64_image
        n_parr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(n_parr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        resistors = detect_resistors(img, app.state.model, app.state.device)
        global_resistance = calculate_global_resistance(resistors)

        frame_id = str(uuid4())

        return {
            "resistors": resistors,
            "global_resistance": global_resistance,
            "image_id": frame_id
        }

    except Exception as e:
        logging.error(f"Error processing live frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": app.state.model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)