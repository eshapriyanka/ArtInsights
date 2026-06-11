import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
from utils.dino_features import extract_dino_feature

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="ArtInsight",
    page_icon="🎨",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Entire App Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e1b4b 50%,
        #312e81 100%
    );
}

/* Main Content */
.main {
    padding-top: 1rem;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 30px 10px;
}

.hero-title {
    font-size: 3.8rem;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    color: #d1d5db;
    font-size: 1.2rem;
}

/* Upload Containers */
[data-testid="stFileUploader"] {
    border: 2px dashed #a78bfa;
    border-radius: 18px;
    padding: 25px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #7c3aed,
        #9333ea
    );
    color: white;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px 20px;
    color: white;
    font-weight: 600;
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Images */
img {
    border-radius: 15px;
}

/* Horizontal Line */
hr {
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_style_model():
    return tf.saved_model.load("style_transfer_model")


@st.cache_resource
def load_svm_model():
    return joblib.load("model/svm_model.pkl")


style_model = load_style_model()
svm_model = load_svm_model()

# ==================================================
# UTILITIES
# ==================================================

def load_image(image_file, max_dim=512):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((max_dim, max_dim))
    img = np.array(img) / 255.0
    return tf.expand_dims(
        tf.convert_to_tensor(img, dtype=tf.float32),
        axis=0
    )


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = tf.cast(tensor, tf.uint8)
    return Image.fromarray(tensor[0].numpy())


# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🎨 ArtInsight</div>
    <div class="hero-subtitle">
        AI-Powered Artwork Authentication & Style Transformation
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs(
    [
        "🎨 Style Transfer",
        "🔍 Artwork Authentication"
    ]
)

# ==================================================
# STYLE TRANSFER
# ==================================================

with tab1:

    st.subheader("Transform Images into Artistic Masterpieces")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        content_img = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="content"
        )

    if content_img:

        st.image(
            content_img,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🎨 Generate Stylized Artwork",
            key="stylize_btn"
        ):

            with st.spinner("Applying artistic style..."):

                content_tensor = load_image(content_img)

                output = style_model(
                    tf.constant(content_tensor),
                    tf.constant(content_tensor)
                )

                stylized_img = tensor_to_image(output[0])

            st.success("Artwork generated successfully!")

            st.markdown("### Results")

            left, right = st.columns(2)

            with left:
                st.image(
                    content_img,
                    caption="Original Image",
                    use_container_width=True
                )

            with right:
                st.image(
                    stylized_img,
                    caption="Stylized Artwork",
                    use_container_width=True
                )

# ==================================================
# ARTWORK AUTHENTICATION
# ==================================================

with tab2:

    st.subheader("Detect AI-Generated Artworks")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        uploaded_file = st.file_uploader(
            "Upload artwork",
            type=["jpg", "jpeg", "png"],
            key="auth"
        )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Artwork",
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Artwork",
            key="detect_btn"
        ):

            with st.spinner(
                "Extracting DINOv2 features and analyzing..."
            ):

                features = extract_dino_feature(
                    image
                ).reshape(1, -1)

                prob = svm_model.predict_proba(features)[0]
                pred = svm_model.predict(features)[0]

            confidence = float(prob[pred]) * 100

            st.markdown("### Analysis Result")

            if pred == 0:

                st.success(
                    f"🧠 Human-made Artwork"
                )

            else:

                st.error(
                    f"🤖 AI-generated Artwork"
                )

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(prob[pred]))

            st.markdown("### Prediction Breakdown")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Human-made",
                    f"{prob[0]*100:.2f}%"
                )

            with c2:
                st.metric(
                    "AI-generated",
                    f"{prob[1]*100:.2f}%"
                )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Built with TensorFlow, DINOv2, Scikit-learn and Streamlit"
)