import streamlit as st
import requests
from PIL import Image
import numpy as np
from image_prediction import create_image
import cv2
import numpy as np

# Set page tab display
st.set_page_config(
    page_title="Simple Image Uploader",
    page_icon="🖼",
    layout="wide",
    initial_sidebar_state="expanded",
)

img_file_buffer = st.camera_input("")

if img_file_buffer is not None:
    col1, col2 = st.columns(2)

    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    res = requests.post(
        url="https://face-tally-r5t56frjwa-no.a.run.app/upload_image",
        files={"img": bytes_data},
    ).json()["boundsboxes"]

    array_original_image = np.array(Image.open(img_file_buffer))

    created_image = create_image(
        array_original_image, res
    )  # Assuming create_image is defined somewhere
    count = len(res)

    with col1:
        # Display the image user uploaded
        st.markdown("Results from the face classifier👇")
        st.write("Tally: **" + str(count) + "** " + ("face" if count == 1 else "faces"))
        st.image(Image.fromarray(created_image))
