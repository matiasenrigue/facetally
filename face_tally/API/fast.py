from face_tally.ml_logic.bound_boxes import getting_bounding_boxes, create_image

# Luego hay que display la imagen nueva
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import requests
from io import BytesIO
import cv2
import datetime
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response




app = FastAPI()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"status": "ok"}




@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    model = YOLO("yolov8n.pt")
    register_heif_opener()
    contents = await img.read()

    image = Image.open(contents)
    boundsboxes = getting_bounding_boxes(image, model)

    return dict(boundsboxes)
