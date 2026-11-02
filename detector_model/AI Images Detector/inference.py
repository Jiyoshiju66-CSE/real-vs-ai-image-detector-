import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import io

from model.convnext import build_model
from transforms import test_transforms
from streamlit_paste_button import paste_image_button as pbutton

MAX_SIZE_MB = 50
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
MAX_PIXELS = 4096 * 4096

st.set_page_config(page_title="Toufik AI Gen & DeepFake Images Detector", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource # for persistance when rerun
def load_model():
    model = build_model()
    ckpt = torch.load("checkpoints/checkpoint_phase2.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model

def safe_open_uploaded(uploaded_file): # validate file + reset cursor 
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError):
        return None

def safe_open_pasted(pil_img): # normalize paste image into rgb reliably
    try:
        return pil_img.convert("RGB")
    except Exception:
        return None

model = load_model()
transform = test_transforms()

st.title("Toufik's AI Generated & DeepFake Images Detector")
st.write("Upload or paste an image in the left sidebar and the model will try to predict **Real vs Fake**.")

# sidebar
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    max_upload_size=MAX_SIZE_MB  
)
with st.sidebar:
    paste_result = pbutton("Paste Image")

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fake threshold", 0.0, 1.0, 0.50, 0.01)
st.sidebar.caption("Higher = stricter (more fake confidence to label Fake)")


image = None
if uploaded_file is not None: # uploaded image stats
    if uploaded_file.size > MAX_SIZE_BYTES:
        st.error(f"File too large. Maximum allowed size is {MAX_SIZE_MB}MB.")
        st.stop()
    image = safe_open_uploaded(uploaded_file)
    if image is None:
        st.error("Invalid image file.")
        st.stop()

elif paste_result.image_data is not None:
    buffer = io.BytesIO()
    paste_result.image_data.save(buffer, format="PNG")
    if buffer.tell() > MAX_SIZE_BYTES:
        st.error(f"Pasted image too large. Maximum allowed size is {MAX_SIZE_MB}MB.")
        st.stop()
    image = safe_open_pasted(paste_result.image_data)
    if image is None:
        st.error("Invalid pasted image.")
        st.stop()

# safety
if image is not None and (image.width * image.height > MAX_PIXELS):
    st.error(f"Image resolution too large. Please upload <= {MAX_PIXELS} pixels.")
    st.stop()

# inference (with atomic rendering to avoid bugs)
result_box = st.container()
if image is None:
    with result_box:
        st.markdown(
                "<div style='height:420px; display:flex; align-items:center; justify-content:center; "
                "background:#111; border-radius:10px; color:#888;'>"
                "Image preview will appear here"
                "</div>",
                unsafe_allow_html=True)
else:
    with result_box:
        preview = image.copy()
        preview.thumbnail((720, 420))
        st.image(preview, caption="Input Image")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]

        real_prob = probs[0].item()
        fake_prob = probs[1].item()

        is_fake = fake_prob > threshold
        label = "Fake" if is_fake else "Real"
        confidence = fake_prob if is_fake else real_prob

        if label == "Fake":
            st.error(f"**{label}**")
        else:
            st.success(f"**{label}**")

        st.metric("Confidence", f"{confidence*100:.2f}%", border=True)
        st.progress(confidence)

        st.caption("Model probabilities")
        st.write(f"Real: **{real_prob*100:.2f}%**")
        st.write(f"Fake: **{fake_prob*100:.2f}%**")