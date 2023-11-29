import streamlit as st
import requests
from PIL import Image
import numpy as np

from face_tally.ml_logic.image_prediction import create_image
from pillow_heif import register_heif_opener


# This is the frontend for our API:
# - You can upload a picture that will be sent to an API
# - From that API it will receive a prediction concerning the bounding boxes
# - It will put togheter those bounding boxes & the original picture and create a final image with both


# Set page tab display
st.set_page_config(
    page_title="Simple Image Uploader",
    page_icon="🖼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Create a native Streamlit file upload input
img_file_buffer = st.file_uploader("Upload an image")

# This is given to the code to give Python the ability to read iPhone pictures
register_heif_opener()


if img_file_buffer is not None:
    col1, col2 = st.columns(2)

    img_bytes = img_file_buffer.getvalue()

    res = requests.post(
        url="http://127.0.0.1:8001/upload_image", files={"img": img_bytes}
    ).json()["boundsboxes"]

    # Things done in the API:
    # - model = YOLO("yolov8n.pt")
    # - image = Image.open(img_file_buffer)
    # - boundsboxes = getting_bounding_boxes(image, model)

    array_original_image = np.array(Image.open(img_file_buffer))

    created_image = create_image(array_original_image, res)

    with col1:
        ### Display the image user uploaded
        st.markdown("Here are the faces in the image you uploaded👇")
        st.image(Image.fromarray(created_image), caption="You can now save your image")
