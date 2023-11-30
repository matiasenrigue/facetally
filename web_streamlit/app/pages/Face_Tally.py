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


# Custom color theme
primary_color = "#F63366"
background_color = "#FFFFFF"
secondary_background_color = "#F0F2F6"
text_color = "#262730"
font = "sans serif"

# Apply the color theme
css = f"""
    <style>
        body {{
            color: {text_color};
            background-color: {background_color};
            font-family: {font};
        }}
        .stApp {{
            background-color: {secondary_background_color};
        }}
        .stTextInput, .stTextArea, .stSelectbox, .stSlider, .stNumberInput, .stCheckbox {{
            background-color: {secondary_background_color};
        }}
        .stButton, .stFileUploader, .stDownloadButton, .stDeckGlJson {{
            background-color: {primary_color};
            color: {background_color};
        }}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Large, stylized title
st.title("Let's go live! 📸")

# Create a native Streamlit file upload input
img_file_buffer = st.file_uploader("Test Face Tally on your best pics")

# This is given to the code to give Python the ability to read iPhone pictures
register_heif_opener()


if img_file_buffer is not None:
    col1, col2 = st.columns(2)

    img_bytes = img_file_buffer.getvalue()

    res = requests.post(
        url="https://face-tally-r5t56frjwa-no.a.run.app/upload_image",
        files={"img": img_bytes},
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
