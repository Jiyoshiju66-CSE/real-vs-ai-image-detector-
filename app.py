
import sys
import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# 1. PROJECT AND MODEL PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "detector_model" / "AI Images Detector"
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "checkpoint_phase2.pth"

if not MODEL_DIR.exists():
    st.error(f"Model folder not found: {MODEL_DIR}")
    st.stop()

sys.path.insert(0, str(MODEL_DIR))

from model.convnext import build_model
from transforms import test_transforms

# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ImageGuard AI",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 ImageGuard AI")
st.subheader("AI-Generated Image Detector")

st.write(
    "Upload an image to estimate whether it is likely to be "
    "real or AI-generated using a pretrained deep-learning model."
)

# --------------------------------------------------
# 3. DEVICE SELECTION
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

st.caption(f"Compute device: {DEVICE}")

# --------------------------------------------------
# 4. LOAD AND CACHE THE PRETRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_detector():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = build_model()

    # Load checkpoint safely onto CPU first
    checkpoint = torch.load(
        str(CHECKPOINT_PATH),
        map_location="cpu"
    )

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# 5. CACHE IMAGE PREPROCESSING
# --------------------------------------------------

@st.cache_resource
def load_transform():
    return test_transforms()


# --------------------------------------------------
# 6. IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # Limit upload size to 50 MB
    MAX_SIZE_BYTES = 50 * 1024 * 1024

    if uploaded_file.size > MAX_SIZE_BYTES:
        st.error("Image is too large. Maximum file size is 50 MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("Invalid image. Please upload a valid image file.")
        st.stop()

    # Limit image resolution
    if image.width * image.height > 4096 * 4096:
        st.error("Image resolution is too large to process.")
        st.stop()

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    st.write(f"Image dimensions: {image.width} × {image.height}")

    # --------------------------------------------------
    # 7. RUN THE PRETRAINED DETECTOR
    # --------------------------------------------------

    try:
        start_time = time.perf_counter()

        with st.spinner(
            "Analyzing image... The first prediction may take longer."
        ):
            # These functions reuse cached resources after first load
            model = load_detector()
            transform = load_transform()

            image_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits = model(image_tensor)
                probabilities = F.softmax(logits, dim=1)[0]

            # Ensure GPU operations are finished before timing ends
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            elif DEVICE.type == "cuda":
                torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time

        # The official model code uses index 0 = Real, index 1 = Fake
        real_probability = probabilities[0].item()
        fake_probability = probabilities[1].item()

        # --------------------------------------------------
        # 8. DISPLAY RESULTS
        # --------------------------------------------------

        st.divider()
        st.subheader("Detection Results")

        if fake_probability > 0.5:
            st.error("Prediction: Likely AI-generated")
            confidence = fake_probability
        else:
            st.success("Prediction: Likely real")
            confidence = real_probability

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Real probability",
                f"{real_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "AI-generated probability",
                f"{fake_probability * 100:.2f}%"
            )

        st.write("Model confidence for its selected prediction")
        st.progress(max(0.0, min(1.0, confidence)))

        st.caption(f"Inference time: {elapsed_time:.2f} seconds")

        st.info(
            "These scores are model estimates, not guarantees. "
            "The detector can misclassify real images and images "
            "created by unfamiliar AI generators."
        )

    except Exception as error:
        st.error("The detector could not analyze this image.")
        st.exception(error)

else:
    st.info("Upload an image to begin detection.")

    st.markdown(
        """
        **Supported formats:** JPG, JPEG, PNG, WEBP

        **How it works**
        1. Upload an image.
        2. The pretrained model analyzes its visual features.
        3. ImageGuard AI displays the prediction and probability scores.
        """
    )
