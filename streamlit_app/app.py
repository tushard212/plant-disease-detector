import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import efficientnet.tfkeras as efn
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, 'assets/images/logo.png')

st.set_page_config(
    page_title='Virtue · Plant Disease Detector',
    page_icon=LOGO_PATH,
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ---------------- Helpers ----------------
def _img_to_b64(path):
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

LOGO_B64 = _img_to_b64(LOGO_PATH)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px;}

    /* Page background */
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #143d2b 0%, transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, #1f3b4d 0%, transparent 55%),
                    linear-gradient(180deg, #0c0f14 0%, #0a0d12 100%);
        color: #e7ecef;
    }

    /* Navbar */
    .virtue-nav {
        display:flex; align-items:center; justify-content:space-between;
        padding: 14px 22px; border-radius: 14px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 28px;
    }
    .virtue-brand {display:flex; align-items:center; gap:12px; font-weight:800; font-size:1.35rem; letter-spacing:.5px;}
    .virtue-brand img {height:38px; width:38px; border-radius:8px;}
    .virtue-brand span {background: linear-gradient(90deg,#7bed9f,#70a1ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
    .virtue-links a {color:#cfd6dc; text-decoration:none; margin-left:18px; font-weight:500; font-size:.95rem;}
    .virtue-links a:hover {color:#7bed9f;}

    /* Hero */
    .hero {padding: 28px 8px 18px 8px;}
    .hero h1 {font-size: 2.6rem; line-height:1.15; margin:0 0 10px 0; font-weight:800;}
    .hero h1 .accent {background: linear-gradient(90deg,#7bed9f,#70a1ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
    .hero p {color:#aab4bd; font-size:1.05rem; max-width:720px;}

    /* Feature cards */
    .features {display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin: 22px 0 8px 0;}
    .feat {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius:14px; padding:18px;
    }
    .feat .ico {font-size:1.4rem; margin-bottom:6px;}
    .feat h4 {margin:6px 0 4px 0; font-size:1.05rem;}
    .feat p {color:#9aa3ad; font-size:.88rem; margin:0;}
    @media (max-width: 800px){ .features{grid-template-columns:1fr;} }

    /* Upload card */
    .upload-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius:16px; padding: 18px 20px; margin-top:10px;
    }
    .upload-card h3 {margin:0 0 6px 0;}
    .upload-card .hint {color:#9aa3ad; font-size:.9rem; margin-bottom:10px;}

    /* File uploader tweaks */
    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px dashed rgba(123,237,159,0.35) !important;
        border-radius:12px !important;
    }

    /* Result tiles */
    .tile-ok    {background: rgba(46,213,115,0.10); border:1px solid rgba(46,213,115,0.35); color:#7bed9f;}
    .tile-bad   {background: rgba(255,71,87,0.10);  border:1px solid rgba(255,71,87,0.35);  color:#ff6b81;}
    .res-tile {padding:14px 16px; border-radius:12px; font-weight:600; margin:14px 0;}

    .sol-card {
        background: rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:12px; padding:14px 16px; margin-top:8px;
    }
    .sol-card h4 {margin:0 0 8px 0; color:#70a1ff;}
    .sol-card ul {margin:0; padding-left:18px;} .sol-card li {margin:4px 0;}

    /* Footer */
    .virtue-foot {
        margin-top:36px; padding:18px; text-align:center;
        color:#7e8a94; border-top:1px solid rgba(255,255,255,0.07);
        font-size:.88rem;
    }
    .virtue-foot a {color:#70a1ff; text-decoration:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Navbar ----------------
logo_img = f'<img src="data:image/png;base64,{LOGO_B64}" alt="logo">' if LOGO_B64 else ''
st.markdown(
    f"""
    <div class="virtue-nav">
      <div class="virtue-brand">{logo_img}<span>Virtue</span></div>
      <div class="virtue-links">
        <a href="#detector">Detector</a>
        <a href="#how">How it works</a>
        <a href="https://github.com/tushard212/plant-disease-detector" target="_blank">GitHub</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Hero ----------------
st.markdown(
    """
    <div class="hero">
      <h1>Keep your green surroundings <span class="accent">safe</span>.</h1>
      <p>Upload a leaf image and Virtue will tell you if your plant is healthy or showing signs of <b>rust</b>, <b>scab</b>, or <b>multiple diseases</b> — with suggested treatment steps.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Features ----------------
st.markdown(
    """
    <div class="features">
      <div class="feat"><div class="ico">⚡</div><h4>Easy to use</h4><p>Just drop a leaf photo — no setup needed.</p></div>
      <div class="feat"><div class="ico">🎯</div><h4>Highly accurate</h4><p>EfficientNet model trained on labeled plant data.</p></div>
      <div class="feat"><div class="ico">🌿</div><h4>Actionable</h4><p>Get clear next-step recommendations.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Detector ----------------
st.markdown('<div id="detector"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="upload-card">
      <h3>🔬 Try the detector</h3>
      <div class="hint">PNG or JPG · up to 200 MB</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))

model = load_plant_model()

uploaded_file = st.file_uploader('Choose your image', type=['png', 'jpg', 'jpeg'], label_visibility='collapsed')

predictions_map = {
    0: 'is healthy',
    1: 'has Multiple Diseases',
    2: 'has rust (Fungus)',
    3: 'has scab (Bacterial)',
}

predictions_sol_rust = ['Choose resistant varieties', 'Keep leaves dry', 'Clean up debris', 'Use fungicides']
predictions_sol_scap = ['Choose resistant varieties', 'Maintain good sanitation', 'Water at the base', 'Use copper-based fungicides']
predictions_sol_vast = [
    'Choose disease-resistant varieties',
    'Practice good sanitation',
    'Water appropriately',
    'Rotate crops — planting different crops in different areas each year helps prevent soil-borne diseases.',
]


def highlight_defects(pil_image):
    """Draw red rectangles around small defect-sized contours on the leaf."""
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10 or w > 100 or h > 100:
            continue
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


def render_solutions(title, items):
    bullets = ''.join(f'<li>{x}</li>' for x in items)
    st.markdown(
        f'<div class="sol-card"><h4>💡 {title}</h4><ul>{bullets}</ul></div>',
        unsafe_allow_html=True,
    )


if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Uploaded leaf**')
        st.image(image, use_column_width=True)

    resized_image = np.array(image.resize((512, 512))) / 255.0
    image_batch = resized_image[np.newaxis, :, :, :]
    predictions_arr = model.predict(image_batch)
    predictions = int(np.argmax(predictions_arr))
    confidence = int(predictions_arr[0][predictions] * 100)

    result_text = f'The plant leaf {predictions_map[predictions]} — {confidence}% confidence'

    if predictions == 0:
        with col2:
            st.markdown('**Result**')
            st.markdown(f'<div class="res-tile tile-ok">✅ {result_text}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sol-card"><h4>🌱 No treatment required</h4><p>Your plant looks healthy — keep up the good care!</p></div>', unsafe_allow_html=True)
    else:
        with col2:
            st.markdown('**Result**')
            st.markdown(f'<div class="res-tile tile-bad">⚠️ {result_text}</div>', unsafe_allow_html=True)
            if predictions == 1:
                render_solutions('Recommended actions (Multiple Diseases)', predictions_sol_vast)
            elif predictions == 2:
                render_solutions('Recommended actions (Rust)', predictions_sol_rust)
            else:
                render_solutions('Recommended actions (Scab)', predictions_sol_scap)

        st.markdown('**Affected regions detected**')
        st.image(highlight_defects(image), use_column_width=True)

# ---------------- How it works ----------------
st.markdown('<div id="how"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="upload-card" style="margin-top:24px;">
      <h3>🛠️ How it works</h3>
      <ol style="color:#cfd6dc; margin:8px 0 0 18px;">
        <li>Upload a leaf image (PNG / JPG).</li>
        <li>The image is resized to 512×512 and normalized.</li>
        <li>An EfficientNet model classifies it into healthy / rust / scab / multiple diseases.</li>
        <li>OpenCV highlights affected regions and suggested actions are shown.</li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Footer ----------------
st.markdown(
    """
    <div class="virtue-foot">
      Built with ❤️ using Streamlit · TensorFlow · OpenCV ·
      <a href="https://github.com/tushard212/plant-disease-detector" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

