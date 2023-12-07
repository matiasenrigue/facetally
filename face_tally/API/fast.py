from face_tally.ml_logic.image_prediction import predict_bounding_boxes
from face_tally.ml_logic.model import get_model
from face_tally.params import *
from face_tally.ml_logic.cards import  image_process
from face_tally.ml_logic.data import update_template_images_from_GCP

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from starlette.responses import Response


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
    await update_template_images_from_GCP()


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



@app.post("/card")
async def card_image(img: UploadFile = File(...)):
    # Receive the image from Streamlit
    contents = await img.read()

    # Open the image from Bytes format
    image = Image.open(BytesIO(contents))
    character_array = np.array(image)

    model = app.state.model

    text_img = image_process(model, character_array)

    return Response(content=text_img.tobytes(), media_type="image/jpg")
