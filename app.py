import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model_architecture import get_model_instance
import os

# ============================================================
# CONFIGURATION DES CLASSES (9 CLASSES – ALIGNÉ AVEC LE MODÈLE)
# ============================================================
CLASS_NAMES = [
    "good",
    "bent_wire",
    "cable_swap",
    "combined",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "missing_cable",
    "missing_wire",
    "poke_insulation"
]

# ===================== COULEURS =============================
PRIMARY_COLOR = "#6C63FF"
SECONDARY_COLOR = "#4ECDC4"
ACCENT_COLOR = "#FF6B9D"
BACKGROUND_DARK = "#1a1a2e"
BACKGROUND_LIGHT = "#16213e"
TEXT_COLOR = "#eaeaea"
SUCCESS_COLOR = "#00d9ff"
DANGER_COLOR = "#ff006e"
GRADIENT_START = "#667eea"
GRADIENT_END = "#764ba2"

# ===================== CONFIG ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_EfficientNetB0.pth"
SEUIL = 0.5

st.set_page_config(
    page_title="IA Inspection Multi-Défauts",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== STYLE ================================
st.markdown(f"""
    <style>
    /* Import de police moderne */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Style global */
    .stApp {{
        background: linear-gradient(135deg, {BACKGROUND_DARK} 0%, {BACKGROUND_LIGHT} 100%);
        font-family: 'Inter', sans-serif;
    }}
    .type-label {{
        background: rgba(255,0,110,0.2);
        border: 1px solid {DANGER_COLOR};
        border-radius: 5px;
        padding: 3px 8px;
        font-weight: bold;
        display: inline-block;
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


# ===================== PREPROCESS ===========================
def preprocess(img: Image.Image):
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

# ===================== LOAD MODEL ===========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = get_model_instance(MODEL_PATH, DEVICE)
    return model

model = load_model()

# ===================== AFFICHAGE GRID =======================
def display_grid(results):
    if not results:
        st.info("Aucune image à afficher.")
        return

    cols = st.columns(4)
    for i, res in enumerate(results):
        with cols[i % 4]:
            st.image(res["image"], use_container_width=True)

            if res["is_anomaly"]:
                st.markdown(
                    f"<p style='color:{DANGER_COLOR};font-weight:bold;text-align:center;'>🚨 ANOMALIE</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='text-align:center;'><span class='type-label'>{res['nom_defaut']}</span></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='text-align:center;'>Confiance  : {res['score_defaut']:.1%}</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<p style='color:{SUCCESS_COLOR};font-weight:bold;text-align:center;'>✅ NORMAL</p>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<p style='text-align:center;'>Score brute : {res['raw_prob']:.4f}</p>",
                unsafe_allow_html=True
            )
            
            # with st.expander("🛠️ Détails Debug (Probabilités)"):
            #     st.write("Distribution des scores :")
            #     st.json(res["all_probs"])

# ===================== INTERFACE ============================
st.title("🔍 Inspection Qualité par IA")
st.markdown("Analyse **binaire + classification multi-défauts (8 classes)**")

st.sidebar.markdown("### ℹ️ Informations")
st.sidebar.markdown(f"🖥️ Device : **{DEVICE}**")
st.sidebar.markdown("🧠 Modèle : **EfficientNetB0**")
# st.sidebar.markdown(f"🎯 Seuil anomalie : **{SEUIL}**")

files = st.file_uploader(
    "📤 Charger des images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ===================== INFERENCE ============================
if files and model:
    results = []

    with st.spinner("Analyse en cours..."):
        for file in files:
            img = Image.open(file)

            with torch.no_grad():
                # ⚠️ ORDRE CORRECT : (binaire, multi)
                binary_out, multi_out = model(preprocess(img))

                # -------- BINAIRE --------
                probability = binary_out.item()
                is_anomaly = probability > SEUIL

                # -------- MULTI-CLASSE --------
                assert multi_out.shape[1] == len(CLASS_NAMES), (
                    f"Erreur modèle : {multi_out.shape[1]} sorties "
                    f"pour {len(CLASS_NAMES)} classes"
                )

                probs = F.softmax(multi_out, dim=1)
                score_defaut, index_classe = torch.max(probs, dim=1)

                idx = index_classe.item()
                nom_defaut = CLASS_NAMES[idx]

            results.append({
                "image": img,
                "is_anomaly": is_anomaly,
                "raw_prob": probability,
                "nom_defaut": nom_defaut,
                "score_defaut": score_defaut.item(),
                # DEBUG INFO
                "all_probs": {name: probs[0][i].item() for i, name in enumerate(CLASS_NAMES)}
            })

    # ===================== TABS ============================
    tab1, tab2, tab3 = st.tabs(["📊 Toutes", "🚨 Anomalies", "✅ Conformes"])

    with tab1:
        display_grid(results)

    with tab2:
        display_grid([r for r in results if r["is_anomaly"]])

    with tab3:
        display_grid([r for r in results if not r["is_anomaly"]])

# ===================== FOOTER ==============================
st.markdown(
    "<hr><p style='text-align:center;'>Système Intelligent d’Inspection Visuelle | PIC 2026</p>",
    unsafe_allow_html=True
)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
print(checkpoint.keys())  # pour voir les clés
print(checkpoint['multiclass_output.weight'].shape)
print(checkpoint['multiclass_output.bias'].shape)