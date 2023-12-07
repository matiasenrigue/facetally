import av
import requests
import streamlit as st
from image_prediction import create_image
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def __init__(self):
        self.count = 0

    def recv(self, frame):
        format = "bgr24"
        img = frame.to_ndarray(format=format)

        _, encoded_image = cv2.imencode(".jpg", img)
        bytes_data = encoded_image.tobytes()

        res = requests.post(
            url="https://face-tally-r5t56frjwa-no.a.run.app/upload_image",
            files={"img": bytes_data},
        ).json()["boundsboxes"]

        created_image = create_image(
            img, res
        )  # Assuming create_image is defined somewhere

        self.count = len(res)

        return av.VideoFrame.from_ndarray(created_image, format=format)


def main():
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        # mode=WebRtcMode.SENDRECV,
        # rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        # async_processing=False,
    )

    with st.sidebar:
        if st.button("Count"):
            try:
                count = webrtc_ctx.video_transformer.count
                st.write(f'<span style="font-size: 36px;">{count}</span>')
            except:
                st.warning("Please start the video")


if __name__ == "__main__":
    main()
