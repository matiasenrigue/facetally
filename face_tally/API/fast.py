from face_tally.ml_logic.image_prediction import predict_bounding_boxes
from ultralytics import YOLO
from PIL import Image

from pillow_heif import register_heif_opener
from fastapi import FastAPI, UploadFile, File

from io import BytesIO


"""
This script receives an API call:
- a picture in Bytes format.

And outputs a prediction made with the Model, in the form of bounding boxes
"""


app = FastAPI()
app.state.model = YOLO("yolov8n.pt")

register_heif_opener()


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
    boundsboxes = predict_bounding_boxes(image, model)

    # Convert boundsboxes to a JSON-serializable format
    return {"boundsboxes": boundsboxes}
