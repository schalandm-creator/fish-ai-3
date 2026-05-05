import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

# ====================== MODERN PAGE CONFIG ======================
st.set_page_config(
    page_title="FischID • KI Fisch-Erkennung",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== FANCY CUSTOM CSS ======================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
        color: #e2e8f0;
    }
    .stApp h1 {
        font-size: 3.2rem;
        background: linear-gradient(90deg, #60a5fa, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero {
        text-align: center;
        padding: 2rem 0;
    }
    .card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border-radius: 12px;
        height: 3rem;
        font-weight: bold;
    }
    .result-card {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown('<div class="hero"><h1>🐟 FischID</h1><p style="font-size:1.3rem; color:#94a3b8;">KI-Fisch-Erkennung • Deutschland</p></div>', unsafe_allow_html=True)

# ====================== MODELL & DATA LADEN ======================
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    return model

model = load_model()

CLASS_NAMES = ["Zander", "Flussbarsch", "Hecht", "Meerforelle", "Brassen", "Karpfen", "Aal", "Wels", "Scholle", "Rotauge"]

with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# ====================== CAMERA + UPLOAD ======================
st.subheader("📸 Foto machen oder hochladen")

col_a, col_b = st.columns(2)

with col_a:
    camera_photo = st.camera_input("Mit Kamera direkt aufnehmen", key="camera")

with col_b:
    uploaded_file = st.file_uploader("Oder Foto aus Galerie hochladen", type=["jpg", "jpeg", "png"])

# Welches Bild wird verwendet?
if camera_photo is not None:
    image = Image.open(camera_photo).convert("RGB")
    source = "Kamera"
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source = "Upload"
else:
    image = None

if image is not None:
    st.image(image, caption=f"Aufgenommen über {source}", use_column_width=True)

    # Vorhersage
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 KI analysiert den Fisch..."):
        predictions = model.predict(img_array, verbose=0)[0]

    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx]) * 100
    predicted_fish = CLASS_NAMES[top_idx]

    if confidence >= 85:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {predicted_fish}")
        st.success(f"**Sicherheit:** {confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Bundesland
        bundesland = st.selectbox("🌍 Bundesland auswählen", 
                                  options=list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(predicted_fish, {})

        st.subheader("📜 Fangregelung")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
        with c2:
            st.metric("Schonzeit", info.get('schonzeit', "Keine"))
        with c3:
            st.metric("Art", predicted_fish)

    else:
        st.error(f"❌ Nicht sicher erkannt ({confidence:.1f}%). Bitte neues Foto versuchen.")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
    <p style="text-align:center; color:#64748b;">
    Moderne KI-Fisch-Erkennungs-App • Eigenes trainiertes Modell • 
    Für Schulprojekt entwickelt
    </p>
""", unsafe_allow_html=True)

st.caption("💡 Tipp: Am besten bei gutem Licht und seitlich auf den Fisch fotografieren")
