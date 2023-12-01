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
st.title("witamy fasetowo! 📸")

# Description of the app with a sprinkle of humor
st.write(
    """
    Chwyć swoje wyimaginowane okulary przeciwsłoneczne, bo zaraz wejdziesz do Facetally Funhouse! 🕶️

    Facetally to Twoja ulubiona aplikacja do rozpoznawania twarzy i liczenia tłumów. Zaufaj nam, razem z nami liczymy twarze
    staje się zabawnym i fascynującym doświadczeniem! 🚀

    **Krok w górę!**
    - 🤖 Doświadcz magii precyzyjnego rozpoznawania twarzy, nawet w najbardziej szalonym tłumie.
    - 🕰️ Doświadcz liczenia twarzy w czasie rzeczywistym z wydajnością i precyzją, dostarczając cennych informacji.

    **Odblokuj możliwości:**
    - 🎉 Bezproblemowo zarządzaj tłumami na imprezach i w przestrzeni publicznej, optymalizując efektywność operacyjną.
    - 👨‍💻 Pożegnaj się z „Are you here” Kitta? monit każdego ranka: Face Tally może automatycznie wykryć Twoją śliczną buźkę

    **Uchwycić moment:**
    - 📸 Prześlij zdjęcie i zobacz, jak twarze ożywają dzięki analizie dynamicznej naszej aplikacji.

    Zatem zapnij pasy, prześlij zdjęcie i wyrusz w ekscytującą podróż związaną z rozpoznawaniem twarzy
    """
)

# Line of code to include images

local_image_path = "gabriellemacaire/Downloads/selfie-friends.jpeg"
st.image(local_image_path, caption="Image Caption", use_column_width=True)
