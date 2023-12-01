import streamlit as st

# Set home page display

# Set the page title and icon
st.set_page_config(
    page_title="Facetally - Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set the background color and text color
st.markdown(
    """
    <style>
        body {
            background-color: #F0F2F6; /* Background color */
            color: #262730; /* Text color */
            font-family: 'Roboto', sans-serif; /* Font family - Let's add some fun! */
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Large, stylized title with a touch of humor
st.title("Welcome to Facetally! 📸")
st.title("Prueba! 📸")

# Description of the app with a sprinkle of humor
st.write(
    """
    Grab your imaginary sunglasses because you're about to enter the Facetally Funhouse! 🕶️

    Facetally is your go-to app for face recognition and crowd counting. Trust us, with us, counting faces
    becomes a fun and fascinating experience! 🚀

    **Step Right Up!**
    - 🤖 Witness the magic of precise face recognition, even in the craziest crowds.
    - 🕰️ Experience real-time face counting with efficiency and precision, providing valuable insights.

    **Unlock Possibilities:**
    - 🎉 Seamlessly manage crowds at events and public spaces, optimizing operational efficiency.
    - 👨‍💻 Say goodbye to Kitt's "Are you here?" prompt every morning: Face Tally can automatically spot your pretty face

    **Capture the Moment:**
    - 📸 Upload an image and see faces come to life with our app's dynamic analysis.

    So, fasten your seatbelt, upload an image, and let's embark on this thrilling facial recognition journey! 🥳📷
    """
)

# Line of code to include images

local_image_path = "gabriellemacaire/Downloads/selfie-friends.jpeg"
st.image(local_image_path, caption="Image Caption", use_column_width=True)
