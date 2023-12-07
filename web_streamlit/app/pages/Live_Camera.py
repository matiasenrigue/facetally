from streamlit_webrtc import webrtc_streamer
import av
import cv2
import requests
import streamlit as st
from image_prediction import *


def callback(frame):
    format = "bgr24"
    img = frame.to_ndarray(format=format)

    _, encoded_image = cv2.imencode(".jpg", img)
    bytes_data = encoded_image.tobytes()

    res = requests.post(
        url="https://face-tally-r5t56frjwa-no.a.run.app/upload_image",
        files={"img": bytes_data},
    ).json()["boundsboxes"]

    created_image = create_image(img, res)  # Assuming create_image is defined somewhere

    return av.VideoFrame.from_ndarray(created_image, format=format)


def main():
    # Live camera streamlit object
    # rtc_configuration is needed to run in the cloud
    webrtc_streamer(
        key="facetally",
        video_frame_callback=callback,
        # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )


if __name__ == "__main__":
    main()
