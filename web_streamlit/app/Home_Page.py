import streamlit as st

# Set home page display

# Set the page title and icon
st.set_page_config(
    page_title="FaceTally - Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Large, stylized title with a touch of humor
st.title("Welcome to FaceTally! 📸")

# Description of the app with a sprinkle of humor
st.write(
    """
    📸 Unleash the Power of FaceTally! 🚀

    Facetally is your favorite face recognition and crowd counting app. Trust us, counting faces becomes a fun and fascinating experience with us! 🚀

    ***Step Up Your Game!***

    🔢 Instantly tally up the good vibes in your pics!
    🤖 Experience the magic of precise face recognition, even in the craziest crowds.
    🕰️ Enjoy real-time face counting with efficiency and precision, providing valuable insights.

    Unlock Possibilities:

    🎉 Seamlessly manage crowds at events and in public spaces, optimizing operational efficiency.
    👨‍💻 Say goodbye to the "Are you here" Kit? prompt every morning: FaceTally can automatically detect your lovely face.
    🎉 Perfect for parties, selfies, and capturing those unforgettable moments.

    Get ready to tally, share, and spread the joy! 🎊 #FaceTallyMagic
    """
)
