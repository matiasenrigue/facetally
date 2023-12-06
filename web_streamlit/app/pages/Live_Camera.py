import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import requests
from PIL import Image
from image_prediction import create_image


def callback(frame):
    format = "bgr24"
    img = frame.to_ndarray(format=format)
    bytes_data = img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    res = requests.post(
        url="https://face-tally-r5t56frjwa-no.a.run.app/upload_image",
        files={"img": bytes_data},
    ).json()["boundsboxes"]

    array_original_image = np.array(Image.open(img))

    created_image = create_image(
        array_original_image, res
    )  # Assuming create_image is defined somewhere

    # st.image(
    #     Image.fromarray(created_image), caption="You can now save your image"
    # )  # img = cv2.cvtColor(cv2.Canny(img), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(created_image, format=format)


# Live camera streamlit object
# rtc_configuration is needed to run in the cloud
webrtc_streamer(
    key="facetally",
    video_frame_callback=callback,
    # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
