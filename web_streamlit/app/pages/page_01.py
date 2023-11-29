import streamlit as st
import requests
from PIL import Image
import os
from web_streamlit.params import *

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

    with col1:
        ### Display the image user uploaded
        st.image(
            Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️"
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
