
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ImageGuard AI",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 ImageGuard AI")
st.subheader("AI-Generated Image Detector")

st.write(
    "Upload an image to begin checking whether it may be "
    "AI-generated or camera-captured."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

        st.write(f"Image width: {image.width} pixels")
        st.write(f"Image height: {image.height} pixels")

        st.info(
            "Image uploaded successfully! The AI detection model "
            "will be integrated in the next step."
        )

    except Exception as error:
        st.error(f"Unable to open the image: {error}")
