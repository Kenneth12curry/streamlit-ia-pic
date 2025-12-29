import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model_architecture import get_model_instance
import os

# ========== PALETTE DE COULEURS (MODIFIABLE) ==========
PRIMARY_COLOR = "#6C63FF"      # Violet tech principal
SECONDARY_COLOR = "#4ECDC4"    # Cyan/turquoise
ACCENT_COLOR = "#FF6B9D"       # Rose accent
BACKGROUND_DARK = "#1a1a2e"    # Fond sombre
BACKGROUND_LIGHT = "#16213e"   # Fond carte
TEXT_COLOR = "#eaeaea"         # Texte clair
SUCCESS_COLOR = "#00d9ff"      # Bleu cyan pour normal
DANGER_COLOR = "#ff006e"       # Rose vif pour anomalie
GRADIENT_START = "#667eea"     # Début gradient
GRADIENT_END = "#764ba2"       # Fin gradient

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_EfficientNetB0.pth"
SEUIL = 0.5  # Seuil fixe pour la détection d'anomalies

st.set_page_config(
    page_title="Détections d'Anomalies", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Personnalisé avec thème IA/Data Science ---
st.markdown(f"""
    <style>
    /* Import de police moderne */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Style global */
    .stApp {{
        background: linear-gradient(135deg, {BACKGROUND_DARK} 0%, {BACKGROUND_LIGHT} 100%);
        font-family: 'Inter', sans-serif;
    }}
    
    /* Titres avec effet gradient */
    h1 {{
        background: linear-gradient(90deg, {GRADIENT_START}, {GRADIENT_END});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 1rem;
    }}
    
    h2, h3 {{
        color: {TEXT_COLOR};
        font-weight: 600;
    }}
    
    /* Sidebar stylisée */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BACKGROUND_LIGHT} 0%, {BACKGROUND_DARK} 100%);
        border-right: 2px solid {PRIMARY_COLOR};
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {TEXT_COLOR};
    }}
    
    /* Boutons et sliders */
    .stSlider > div > div > div {{
        background: {PRIMARY_COLOR};
    }}
    
    /* Cartes de résultat améliorées */
    .result-card {{
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }}
    
    .result-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(108, 99, 255, 0.4);
    }}
    
    /* Tabs stylisées */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: {TEXT_COLOR};
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border: none;
    }}
    
    /* Upload zone */
    [data-testid="stFileUploader"] {{
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed {PRIMARY_COLOR};
        border-radius: 15px;
        padding: 20px;
    }}
    
    /* Images avec effet */
    .stImage {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }}
    
    /* Captions */
    .caption {{
        color: {TEXT_COLOR};
        opacity: 0.8;
        font-size: 0.9rem;
    }}
    
    /* Footer moderne */
    .footer {{
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        background: rgba(255, 255, 255, 0.03);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: {TEXT_COLOR};
        font-weight: 300;
        border-radius: 15px;
    }}
    
    /* Info boxes */
    .stAlert {{
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid {SECONDARY_COLOR};
        border-radius: 8px;
    }}
    
    /* Effet glow sur les labels */
    .anomaly-label {{
        text-shadow: 0 0 10px {DANGER_COLOR};
        animation: pulse 2s infinite;
    }}
    
    .normal-label {{
        text-shadow: 0 0 10px {SUCCESS_COLOR};
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    /* Texte général */
    p, span, div {{
        color: {TEXT_COLOR};
    }}
    </style>
""", unsafe_allow_html=True)

# --- Prétraitement (SANS NORMALISATION pour éviter le 1.0000) ---
def preprocess(img):
    img = img.convert('RGB')
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return t(img).unsqueeze(0).to(DEVICE)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return get_model_instance(MODEL_PATH, DEVICE)
    return None

model = load_model()

def display_grid(results):
    if not results:
        st.info("🔍 Aucune image à afficher ici.")
        return
    cols = st.columns(4)
    for idx, res in enumerate(results):
        with cols[idx % 4]:
            color = DANGER_COLOR if res["is_anomaly"] else SUCCESS_COLOR
            label = "🚨 ANOMALIE" if res["is_anomaly"] else "✅ NORMAL"
            label_class = "anomaly-label" if res["is_anomaly"] else "normal-label"
            
            st.image(res["image"], use_container_width=True)
            st.markdown(f"<p class='{label_class}' style='color:{color}; font-weight:bold; text-align:center; font-size:1.1rem;'>{label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='caption' style='text-align:center;'>📊 Confiance : {res['confidence']:.2%}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='caption' style='text-align:center;'>🎯 Score brut : {res['raw_prob']:.4f}</p>", unsafe_allow_html=True)

# --- Interface Principale ---
st.markdown("<br>", unsafe_allow_html=True)
st.title("🔍 Inspection Qualité par IA")
st.markdown(f"<p style='color:{TEXT_COLOR}; font-size:1.2rem; opacity:0.8;'>Détection d'anomalies en temps réel avec Deep Learning</p>", unsafe_allow_html=True)

# --- Sidebar avec informations (SANS le slider) ---
st.sidebar.markdown(f"<h2 style='color:{PRIMARY_COLOR};'>ℹ️ Informations</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='color:{TEXT_COLOR};'>🖥️ Device: <b>{DEVICE}</b></p>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{TEXT_COLOR};'>🧠 Modèle: <b>EfficientNetB0</b></p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='color:{TEXT_COLOR}; opacity:0.7; font-size:0.85rem;'>💡 Le modèle analyse les images en temps réel pour détecter les anomalies de qualité.</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
files = st.file_uploader("📤 Charger des photos", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if files and model:
    all_results = []
    
    with st.spinner('🔄 Analyse en cours...'):
        for file in files:
            img = Image.open(file)
            with torch.no_grad():
                binary_out, _ = model(preprocess(img))
                probability = binary_out.item()
                is_anomaly = probability > SEUIL
                confidence = probability if is_anomaly else (1 - probability)
                
                all_results.append({
                    "name": file.name, "image": img, "is_anomaly": is_anomaly,
                    "confidence": confidence, "raw_prob": probability
                })

    # --- Affichage des résultats ---
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Vue Globale", "🚨 Anomalies", "✅ Conformes"])
    
    with tab1:
        st.markdown(f"<h3 style='color:{TEXT_COLOR};'>Toutes les images analysées</h3>", unsafe_allow_html=True)
        display_grid(all_results)
    
    with tab2:
        anomalies = [r for r in all_results if r["is_anomaly"]]
        st.markdown(f"<h3 style='color:{DANGER_COLOR};'>Anomalies détectées ({len(anomalies)})</h3>", unsafe_allow_html=True)
        display_grid(anomalies)
    
    with tab3:
        conformes = [r for r in all_results if not r["is_anomaly"]]
        st.markdown(f"<h3 style='color:{SUCCESS_COLOR};'>Images conformes ({len(conformes)})</h3>", unsafe_allow_html=True)
        display_grid(conformes)

st.markdown(f'''
    <div class="footer">
        <p style='font-size:1.1rem; margin-bottom:10px;'>🤖 Propulsé par <b>EfficientNetB0</b></p>
        <p style='opacity:0.7;'>Analyse en temps réel | Deep Learning | Computer Vision</p>
    </div>
''', unsafe_allow_html=True)