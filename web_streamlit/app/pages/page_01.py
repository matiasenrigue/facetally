import streamlit as st
import requests
from PIL import Image
import os
from web_streamlit.params import *

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




# Set page tab display
st.set_page_config(
    page_title="Simple Image Uploader",
    page_icon="🖼",
    layout="wide",
    initial_sidebar_state="expanded",
)

url = FACETALLY_API_URL
st.text(f"the API url is: {url}")

# App title and description
st.header("Simple Image Uploader ")
st.markdown(
    """
            > * [FastAPI](https://fastapi.tiangolo.com/) on the backend
            > * [PIL/pillow](https://pillow.readthedocs.io/en/stable/) and [opencv-python](https://github.com/opencv/opencv-python) for working with images
            """
)

st.markdown("---")

### Create a native Streamlit file upload input
img_file_buffer = st.file_uploader("Upload an image")



if img_file_buffer is not None:
    col1, col2 = st.columns(2)
    model = YOLO("yolov8n.pt")
    register_heif_opener()
    image = Image.open(img_file_buffer)

    boundsboxes = getting_bounding_boxes(image, model)

    array_original_image = np.array(image)

    created_image = create_image(array_original_image, boundsboxes)

    with col1:
        ### Display the image user uploaded
        st.markdown("Here are the faces in the image you uploaded👇")
        st.image(
            Image.fromarray(created_image), caption="You can now save your image"
        )

    # with col2:
    #     with st.spinner("Wait for it..."):
    #         ### Get bytes from the file buffer
    #         img_bytes = img_file_buffer.getvalue()

    #         ### Make request to  API (stream=True to stream response as bytes)
    #         res = requests.post(url + "/upload_image", files={"img": img_bytes})

    #         if res.status_code == 200:
    #             # Merge the bbox with the original image

    #             # Display the merged image
    #             st.image(res.content, caption="Image returned from API ☝️")

    #         else:
    #             st.markdown("**Oops**, something went wrong 😓 Please try again.")
    #             print(res.status_code, res.content)
