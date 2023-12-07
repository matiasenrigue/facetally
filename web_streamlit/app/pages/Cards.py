import streamlit as st
import requests
from PIL import Image
from image_prediction import create_image
from io import BytesIO

# Set page tab display
st.set_page_config(
    page_title="Simple Image Uploader",
    page_icon="🖼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Large, stylized title
st.title("Energy! 📸")

# img_file_buffer = st.file_uploader("Test Face Tally on your best pics")
img_file_buffer = st.camera_input("Test Energy on your best pics")

url = "https://face-tally-r5t56frjwa-no.a.run.app/card"
# url = "http://127.0.0.1:8002/card"

if img_file_buffer is not None:
    col1, col2 = st.columns(2)

    with col1:
        ### Display the image user uploaded
        st.image(
            Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️"
        )

    with col2:
        with st.spinner("Wait for it..."):
            ### Get bytes from the file buffer
            img_bytes = img_file_buffer.getvalue()

            ### Make request to  API (stream=True to stream response as bytes)
            res = requests.post(url, files={"img": img_bytes})

            if res.status_code == 200:
                # Bytes of the image
                image_bytes = res.content

                # Convert the bytes to a BytesIO object
                bytes_io = BytesIO(image_bytes)

                # Attempt to open the image using PIL
                image = Image.open(bytes_io)

                # Display the image using Streamlit
                st.image(image, caption="Reconstructed Image")

            else:
                st.markdown("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)
