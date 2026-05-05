import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🐟 Fisch-Erkennung Deutschland",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS für schönes Design ======================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp h1 {
        color: #1e3a8a;
        text-align: center;
        font-size: 2.8rem;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 10px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 10px;
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== TITLE & HEADER ======================
st.title("🐟 Deutsche Fisch-Erkennungs-App")
st.markdown("**KI-gestützte Erkennung + aktuelle Fangregelungen**")

st.markdown("---")

# ====================== MODELL LADEN ======================
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_fish_model():
    model_path = "keras_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ keras_model.h5 nicht gefunden!")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.stop()

model = load_fish_model()

# ====================== KLASSENNAMEN ======================
CLASS_NAMES = [
    "Zander", "Flussbarsch", "Hecht", "Meerforelle", "Brassen",
    "Karpfen", "Aal", "Wels", "Scholle", "Rotauge"
]

# ====================== FISH DATA ======================
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("""
    Diese App erkennt 10 häufige Fischarten mit einem eigenen trainierten KI-Modell.  
    Danach werden die aktuellen Schonzeiten und Mindestmaße des gewählten Bundeslandes angezeigt.
    """)
    st.info("**Hinweis:** Die Angaben dienen nur zu Informationszwecken. Prüfe immer die offizielle Verordnung!")

# ====================== HAUPTBEREICH ======================
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("📸 Foto des Fisches hochladen", 
                                     type=["jpg", "jpeg", "png"], 
                                     help="Am besten ein klares Seitenfoto des ganzen Fisches")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Bild anzeigen
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🤖 KI analysiert den Fisch..."):
        predictions = model.predict(img_array, verbose=0)[0]

    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx]) * 100
    predicted_fish = CLASS_NAMES[top_idx]

    if confidence >= 85.0:
        st.success(f"**Erkannte Fischart:** {predicted_fish}  \n**Sicherheit:** {confidence:.1f}% ✅")
        
        # Bundesland Auswahl
        bundesland = st.selectbox(
            "🌍 Aus welchem Bundesland kommst du?", 
            options=list(fish_data["bundeslaender"].keys()),
            key="bundesland_select"
        )
        
        info = fish_data["bundeslaender"][bundesland].get(predicted_fish, {})
        
        # Schöne Anzeige der Ergebnisse
        st.subheader("📏 Fangregelung")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("**Mindestmaß**", f"{info.get('mindestmass', 0)} cm")
        with c2:
            schonzeit = info.get('schonzeit', "Keine")
            st.metric("**Schonzeit**", schonzeit if schonzeit != "Keine" else "Ganzjährig erlaubt")
        
        if schonzeit == "Keine":
            st.caption("Für diese Art gilt in diesem Bundesland keine Schonzeit.")
            
    else:
        st.error(f"❌ Der Fisch konnte nicht sicher erkannt werden ({confidence:.1f}%).")
        st.warning("Tipp: Mach ein klareres Foto mit gutem Licht und zeige den ganzen Fisch von der Seite.")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
**Diese App wurde im Rahmen eines Schulprojekts entwickelt.**  
KI-Modell: Eigenes trainiertes Keras-Modell  
Daten: Zusammenfassung der Landesfischereiverordnungen (Stand 2026)
""")

st.caption("Viel Erfolg beim Angeln – und immer die Regeln beachten! 🐟")
