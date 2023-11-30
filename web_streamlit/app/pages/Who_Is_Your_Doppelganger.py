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
st.title("Find your celebrity doppelganger 📸")
