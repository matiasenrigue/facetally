from face_tally.ml_logic.image_prediction import predict_bounding_boxes
from face_tally.ml_logic.model import get_model
from face_tally.params import *

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from io import BytesIO

"""
This script receives an API call:
- a picture in Bytes format.

And outputs a prediction made with the Model, in the form of bounding boxes
"""


app = FastAPI()


async def load_model():
    model, _ = await get_model(MODEL_SOURCE)
    return model


@app.on_event("startup")
async def startup_event():
    app.state.model = await load_model()


@app.get("/ok")
def read_root():
    return {"status": "ok"}


@app.post("/upload_image")
async def receive_image(img: UploadFile = File(...)):
    # Receive the image from Streamlit
    contents = await img.read()

    # Open the image from Bytes format
    image = Image.open(BytesIO(contents))

    model = app.state.model

    # Process the image using the YOLO model
    boundsboxes = predict_bounding_boxes(image, model, MODEL_SOURCE)

    # Convert boundsboxes to a JSON-serializable format
    return {"boundsboxes": boundsboxes}
