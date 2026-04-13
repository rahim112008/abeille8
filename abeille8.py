"""
ApiTrack Pro – Application de gestion apicole professionnelle
Streamlit + Python + SQLite
CORRECTION : Les fonctions ia_analyser_* utilisent maintenant ia_call()
             (multi-fournisseurs) au lieu de forcer Anthropic uniquement.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import json
import os
import datetime
from pathlib import Path

# ── Plotly (graphiques) ──────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go

# ── Folium (cartographie) ────────────────────────────────────────────────────
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

# ── TensorFlow (optionnel – deep learning) ───────────────────────────────────
try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False

# ── SentinelHub (optionnel) ──────────────────────────────────────────────────
try:
    from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest
    SH_OK = True
except ImportError:
    SH_OK = False

# ── Anthropic (IA gratuite via Claude) ───────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

# ── Base64 pour upload images ─────────────────────────────────────────────────
import base64

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ApiTrack Pro",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "apitrack.db"

# ════════════════════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --gold:         #F5A623;
        --gold-light:   #FFD07A;
        --gold-dark:    #C8820A;
        --bg-app:       #0F1117;
        --bg-main:      #161B27;
        --bg-card:      #1E2535;
        --bg-card2:     #252D40;
        --bg-input:     #1A2030;
        --border:       #2E3A52;
        --border-light: #3A4A66;
        --text-primary: #F0F4FF;
        --text-second:  #A8B4CC;
        --text-muted:   #6B7A99;
        --text-label:   #8899BB;
        --green:        #34D399;
        --green-bg:     #0D2A1F;
        --green-border: #1A5C3A;
        --yellow:       #FBD147;
        --yellow-bg:    #2A200A;
        --yellow-border:#4A3A10;
        --red:          #F87171;
        --red-bg:       #2A0D0D;
        --red-border:   #5C1A1A;
        --blue:         #60A5FA;
        --blue-bg:      #0D1A2A;
        --blue-border:  #1A3A5C;
    }

    .stApp {
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1400px;
        background: var(--bg-main) !important;
    }
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stMarkdown p {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] {
        background: #080C14 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: #C8D8F0 !important;
    }
    [data-testid="stSidebar"] button {
        background: transparent !important;
        color: #A8B4CC !important;
        border: none !important;
        text-align: left !important;
        font-size: 0.875rem !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: all 0.15s !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(245,166,35,0.12) !important;
        color: var(--gold-light) !important;
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    h2 { font-size: 1.4rem !important; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 16px; }
    h3 { font-size: 1.05rem !important; color: var(--gold-light) !important; }

    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: 3px solid var(--gold) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--gold-light) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-second) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricDelta"] { color: var(--green) !important; }

    .stButton > button {
        background: var(--gold-dark) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 8px 18px !important;
        letter-spacing: 0.02em !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: var(--gold) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(245,166,35,0.3) !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1.5px solid var(--border-light) !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(245,166,35,0.2) !important;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    .stTextInput label, .stNumberInput label,
    .stTextArea label, .stSelectbox label,
    .stSlider label, .stCheckbox label,
    .stFileUploader label {
        color: var(--text-second) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }

    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1.5px solid var(--border-light) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] p {
        color: var(--text-primary) !important;
    }

    .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    .stDataFrame table {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    .stDataFrame thead th {
        background: var(--bg-card2) !important;
        color: var(--gold-light) !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 10px 12px !important;
    }
    .stDataFrame tbody td {
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 8px 12px !important;
        font-size: 0.875rem !important;
    }
    .stDataFrame tbody tr:hover td {
        background: var(--bg-card2) !important;
    }

    [data-testid="stAlert"],
    .stAlert {
        border-radius: 8px !important;
        border-width: 1px !important;
        padding: 12px 16px !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        background: var(--bg-card) !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 4px 8px 0 !important;
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--text-second) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        background: transparent !important;
        border: none !important;
        padding: 8px 16px !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--gold) !important;
        border-bottom: 2px solid var(--gold) !important;
        font-weight: 600 !important;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--gold-light) !important;
        background: rgba(245,166,35,0.08) !important;
    }
    [data-testid="stTabsContent"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 16px !important;
    }

    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        background: var(--bg-card) !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: var(--bg-card2) !important;
        color: var(--gold-light) !important;
    }
    [data-testid="stExpander"] > div {
        background: var(--bg-card) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--bg-input) !important;
        border: 1.5px dashed var(--border-light) !important;
        border-radius: 8px !important;
        color: var(--text-second) !important;
    }
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        color: var(--text-second) !important;
    }

    [data-testid="stDownloadButton"] button {
        background: var(--bg-card2) !important;
        color: var(--gold-light) !important;
        border: 1px solid var(--gold-dark) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: var(--gold-dark) !important;
        color: #FFFFFF !important;
    }

    .api-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        color: var(--text-primary);
    }
    .api-card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--gold-light);
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    .badge-ok   { background:#0D2A1F; color:#6EE7B7; border:1px solid #1A5C3A; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
    .badge-warn { background:#2A200A; color:#FDE68A; border:1px solid #4A3A10; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
    .badge-crit { background:#2A0D0D; color:#FCA5A5; border:1px solid #5C1A1A; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }

    .api-footer {
        text-align: center;
        font-size: 0.72rem;
        color: var(--text-muted);
        padding: 12px;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-card);
        border-radius: 0 0 8px 8px;
    }

    [data-testid="stFormSubmitButton"] button {
        background: var(--gold-dark) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--gold) !important;
    }

    [data-testid="stProgressBar"] > div {
        background: var(--bg-card2) !important;
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--gold-dark), var(--gold)) !important;
    }

    hr { border-color: var(--border) !important; }
    a { color: var(--gold-light) !important; }
    a:hover { color: var(--gold) !important; }

    code {
        background: var(--bg-card2) !important;
        color: var(--gold-light) !important;
        padding: 1px 6px !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em !important;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-app); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold-dark); }

    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# BASE DE DONNÉES SQLITE
# ════════════════════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS ruches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT NOT NULL,
        race TEXT DEFAULT 'intermissa',
        date_installation TEXT,
        localisation TEXT,
        latitude REAL,
        longitude REAL,
        statut TEXT DEFAULT 'actif',
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE CASCADE,
        date_inspection TEXT NOT NULL,
        poids_kg REAL,
        nb_cadres INTEGER,
        varroa_pct REAL,
        reine_vue INTEGER DEFAULT 1,
        comportement TEXT DEFAULT 'calme',
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS traitements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE CASCADE,
        date_debut TEXT NOT NULL,
        date_fin TEXT,
        produit TEXT,
        pathologie TEXT,
        dose TEXT,
        duree_jours INTEGER,
        statut TEXT DEFAULT 'en_cours',
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS recoltes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE CASCADE,
        date_recolte TEXT NOT NULL,
        type_produit TEXT DEFAULT 'miel',
        quantite_kg REAL,
        humidite_pct REAL,
        ph REAL,
        hda_pct REAL,
        qualite TEXT DEFAULT 'A',
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS morph_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id),
        date_analyse TEXT NOT NULL,
        longueur_aile_mm REAL,
        largeur_aile_mm REAL,
        indice_cubital REAL,
        glossa_mm REAL,
        tomentum INTEGER,
        pigmentation TEXT,
        race_probable TEXT,
        confiance_json TEXT,
        specialisation TEXT,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS zones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT NOT NULL,
        type_zone TEXT DEFAULT 'nectar',
        latitude REAL,
        longitude REAL,
        superficie_ha REAL,
        flore_principale TEXT,
        ndvi REAL,
        potentiel TEXT DEFAULT 'modere',
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        action TEXT NOT NULL,
        details TEXT,
        utilisateur TEXT DEFAULT 'admin'
    );

    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)

    pwd_hash = hashlib.sha256("admin1234".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (username, password_hash, email) VALUES (?, ?, ?)",
              ("admin", pwd_hash, "admin@apitrack.pro"))

    _insert_demo_data(c)

    conn.commit()
    conn.close()


def _insert_demo_data(c):
    c.execute("SELECT COUNT(*) FROM ruches")
    if c.fetchone()[0] > 0:
        return

    ruches_demo = [
        ("Zitoun A", "intermissa", "2023-03-15", "Zone Atlas Nord", 34.88, 1.32, "actif"),
        ("Sahara B", "sahariensis", "2023-04-01", "Zone Jujubiers", 34.85, 1.35, "actif"),
        ("Atlas C", "hybride", "2022-05-20", "Zone Cèdres", 34.90, 1.28, "actif"),
        ("Cedre D", "intermissa", "2023-02-10", "Zone Atlas Sud", 34.82, 1.31, "actif"),
        ("Cedre E", "intermissa", "2024-03-01", "Zone Atlas Nord", 34.89, 1.33, "actif"),
        ("Oued F", "intermissa", "2024-04-15", "Bord Oued", 34.87, 1.30, "actif"),
    ]
    for r in ruches_demo:
        c.execute("INSERT INTO ruches (nom, race, date_installation, localisation, latitude, longitude, statut) VALUES (?,?,?,?,?,?,?)", r)

    today = datetime.date.today()
    inspections_demo = [
        (1, str(today), 28.4, 12, 0.8, 1, "calme", "Excellent couvain"),
        (2, str(today - datetime.timedelta(days=1)), 25.6, 10, 1.2, 1, "calme", "RAS"),
        (3, str(today - datetime.timedelta(days=2)), 22.1, 9, 2.4, 0, "nerveuse", "Reine introuvable"),
        (4, str(today - datetime.timedelta(days=3)), 26.9, 11, 1.1, 1, "très calme", "Top productrice"),
        (6, str(today - datetime.timedelta(days=1)), 19.2, 7, 3.8, 1, "agressive", "Traitement urgent"),
    ]
    for i in inspections_demo:
        c.execute("INSERT INTO inspections (ruche_id,date_inspection,poids_kg,nb_cadres,varroa_pct,reine_vue,comportement,notes) VALUES (?,?,?,?,?,?,?,?)", i)

    recoltes_demo = [
        (1, "2025-03-01", "miel", 48.0, 17.2, 3.8, None, "A"),
        (2, "2025-03-01", "miel", 32.0, 17.8, 3.9, None, "A"),
        (1, "2025-01-15", "pollen", 4.5, None, None, None, "A"),
        (4, "2025-03-15", "gelée royale", 0.6, None, None, 2.1, "A+"),
        (1, "2024-09-01", "miel", 62.0, 17.0, 3.7, None, "A"),
    ]
    for r in recoltes_demo:
        c.execute("INSERT INTO recoltes (ruche_id,date_recolte,type_produit,quantite_kg,humidite_pct,ph,hda_pct,qualite) VALUES (?,?,?,?,?,?,?,?)", r)

    morph_demo = [
        (1, str(today), 9.2, 3.1, 2.3, 6.1, 2, "Noir", "intermissa",
         json.dumps([{"race":"intermissa","confiance":72},{"race":"sahariensis","confiance":18},{"race":"hybride","confiance":8},{"race":"ligustica","confiance":2},{"race":"carnica","confiance":0}]),
         "Production miel + propolis"),
    ]
    for m in morph_demo:
        c.execute("INSERT INTO morph_analyses (ruche_id,date_analyse,longueur_aile_mm,largeur_aile_mm,indice_cubital,glossa_mm,tomentum,pigmentation,race_probable,confiance_json,specialisation) VALUES (?,?,?,?,?,?,?,?,?,?,?)", m)

    zones_demo = [
        ("Forêt chênes-lièges", "nectar+pollen", 34.88, 1.31, 120.0, "Quercus suber", 0.72, "élevé"),
        ("Jujubiers Est", "nectar", 34.86, 1.34, 45.0, "Ziziphus lotus", 0.65, "élevé"),
        ("Lavande Sud", "pollen", 34.83, 1.30, 18.0, "Lavandula stoechas", 0.58, "modéré"),
        ("Romarin Ouest", "nectar+pollen", 34.89, 1.28, 30.0, "Rosmarinus officinalis", 0.61, "modéré"),
    ]
    for z in zones_demo:
        c.execute("INSERT INTO zones (nom,type_zone,latitude,longitude,superficie_ha,flore_principale,ndvi,potentiel) VALUES (?,?,?,?,?,?,?,?)", z)

    journal_demo = [
        ("Initialisation base de données", "Données démo insérées", "système"),
        ("Inspection R07 critique", "Varroa 3.8% — alerte générée", "admin"),
        ("Récolte enregistrée", "48 kg miel toutes fleurs, ruche R01", "admin"),
        ("Morphométrie R01", "intermissa 72% — JSON sauvegardé", "admin"),
    ]
    for j in journal_demo:
        c.execute("INSERT INTO journal (action,details,utilisateur) VALUES (?,?,?)", j)

    c.execute("INSERT OR IGNORE INTO settings VALUES ('rucher_nom','Rucher de l Atlas')")
    c.execute("INSERT OR IGNORE INTO settings VALUES ('localisation','Tlemcen, Algérie')")
    c.execute("INSERT OR IGNORE INTO settings VALUES ('version','2.0.0')")


# ════════════════════════════════════════════════════════════════════════════
# AUTHENTIFICATION
# ════════════════════════════════════════════════════════════════════════════
def check_login(username, password):
    conn = get_db()
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, pwd_hash)
    ).fetchone()
    conn.close()
    return user


def login_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-bottom:24px'>
            <div style='font-size:3rem'>🐝</div>
            <h1 style='font-family:Playfair Display,serif;color:#F0F4FF;font-size:2rem;margin:8px 0 4px'>ApiTrack Pro</h1>
            <p style='color:#A8B4CC;font-size:.9rem'>Gestion apicole professionnelle</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Identifiant", placeholder="admin")
            password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)

        if submitted:
            user = check_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                log_action("Connexion", f"Utilisateur {username} connecté")
                st.rerun()
            else:
                st.error("Identifiants incorrects. (Démo : admin / admin1234)")

        st.markdown("<p style='text-align:center;font-size:.75rem;color:#A8B4CC;margin-top:16px'>admin / admin1234 pour la démo</p>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════
def log_action(action, details="", user=None):
    u = user or st.session_state.get("username", "système")
    conn = get_db()
    conn.execute("INSERT INTO journal (action,details,utilisateur) VALUES (?,?,?)", (action, details, u))
    conn.commit()
    conn.close()


def status_badge(varroa):
    if varroa is None:
        return "N/A"
    if varroa >= 3.0:
        return "🔴 Critique"
    elif varroa >= 2.0:
        return "🟡 Surveiller"
    else:
        return "🟢 Bon"


def get_setting(key, default=""):
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else default


# ════════════════════════════════════════════════════════════════════════════
# MOTEUR IA MULTI-FOURNISSEURS — 100% GRATUITS
# ════════════════════════════════════════════════════════════════════════════

IA_PROVIDERS = {
    "🤖 Claude (Anthropic)": {
        "key":        "anthropic_api_key",
        "env":        "ANTHROPIC_API_KEY",
        "url":        "https://console.anthropic.com",
        "prefix":     "sk-ant-",
        "models":     ["claude-opus-4-5", "claude-haiku-4-5-20251001"],
        "default":    "claude-opus-4-5",
        "quota":      "~5$ crédits offerts · ~500 analyses",
        "vision":     True,
        "type":       "anthropic",
    },
    "🌟 Gemma 4 (Google AI Studio)": {
        "key":        "google_api_key",
        "env":        "GOOGLE_API_KEY",
        "url":        "https://aistudio.google.com/app/apikey",
        "prefix":     "AIzaSy",
        "models":     ["gemini-2.0-flash", "gemma-4-31b-it", "gemma-4-27b-it", "gemini-1.5-flash"],
        "default":    "gemini-2.0-flash",
        "quota":      "Gratuit · 1 500 req/jour · 1M tokens/min",
        "vision":     True,
        "type":       "google",
    },
    "⚡ Groq (Ultra-rapide)": {
        "key":        "groq_api_key",
        "env":        "GROQ_API_KEY",
        "url":        "https://console.groq.com/keys",
        "prefix":     "gsk_",
        "models":     ["llama-3.3-70b-versatile", "llama-4-scout-17b-16e-instruct", "gemma2-9b-it"],
        "default":    "llama-3.3-70b-versatile",
        "quota":      "Gratuit · 30 RPM · 1 000 RPD · 800 tok/s",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://api.groq.com/openai/v1",
    },
    "🔀 OpenRouter (Multi-modèles)": {
        "key":        "openrouter_api_key",
        "env":        "OPENROUTER_API_KEY",
        "url":        "https://openrouter.ai/keys",
        "prefix":     "sk-or-",
        "models":     ["meta-llama/llama-4-maverick:free", "deepseek/deepseek-r1:free",
                       "google/gemma-3-27b-it:free", "mistralai/mistral-7b-instruct:free",
                       "qwen/qwen3-235b-a22b:free"],
        "default":    "meta-llama/llama-4-maverick:free",
        "quota":      "Gratuit · ~50 req/jour · accès 200+ modèles",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://openrouter.ai/api/v1",
    },
    "🇪🇺 Mistral AI (GDPR)": {
        "key":        "mistral_api_key",
        "env":        "MISTRAL_API_KEY",
        "url":        "https://console.mistral.ai/api-keys",
        "prefix":     "",
        "models":     ["mistral-large-latest", "mistral-small-latest", "open-mistral-7b"],
        "default":    "mistral-large-latest",
        "quota":      "Gratuit · 1 req/s · 1 milliard tok/mois",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://api.mistral.ai/v1",
    },
    "🔍 Cohere (RAG/Search)": {
        "key":        "cohere_api_key",
        "env":        "COHERE_API_KEY",
        "url":        "https://dashboard.cohere.com/api-keys",
        "prefix":     "",
        "models":     ["command-r-plus", "command-r", "command-a-03-2025"],
        "default":    "command-r-plus",
        "quota":      "Gratuit · 20 RPM · 1 000 req/mois",
        "vision":     False,
        "type":       "cohere",
    },
    "🇨🇳 Zhipu AI / GLM (Gratuit illimité)": {
        "key":        "zhipu_api_key",
        "env":        "ZHIPU_API_KEY",
        "url":        "https://open.bigmodel.cn/usercenter/apikeys",
        "prefix":     "",
        "models":     ["glm-4v-flash", "glm-4-flash", "glm-4-plus"],
        "default":    "glm-4v-flash",
        "quota":      "Gratuit · Limites non documentées · Vision OK",
        "vision":     True,
        "type":       "openai_compat",
        "base_url":   "https://open.bigmodel.cn/api/paas/v4",
    },
    "🧠 Cerebras (Très rapide)": {
        "key":        "cerebras_api_key",
        "env":        "CEREBRAS_API_KEY",
        "url":        "https://cloud.cerebras.ai/platform",
        "prefix":     "csk-",
        "models":     ["llama-3.3-70b", "qwen3-235b", "llama-4-scout-17b"],
        "default":    "llama-3.3-70b",
        "quota":      "Gratuit · 30 RPM · 14 400 RPD",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://api.cerebras.ai/v1",
    },
    "🤗 Hugging Face (10 000 modèles)": {
        "key":        "hf_api_key",
        "env":        "HF_API_KEY",
        "url":        "https://huggingface.co/settings/tokens",
        "prefix":     "hf_",
        "models":     ["mistralai/Mixtral-8x7B-Instruct-v0.1",
                       "meta-llama/Llama-3.3-70B-Instruct",
                       "Qwen/Qwen2.5-72B-Instruct"],
        "default":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "quota":      "Gratuit · Serverless Inference · modèles <10GB",
        "vision":     False,
        "type":       "huggingface",
    },
    "🐙 GitHub Models (GPT-4o gratuit)": {
        "key":        "github_api_key",
        "env":        "GITHUB_TOKEN",
        "url":        "https://github.com/settings/tokens",
        "prefix":     "github_pat_",
        "models":     ["openai/gpt-4o", "openai/gpt-4.1",
                       "meta-llama/Llama-3.3-70B-Instruct",
                       "deepseek/DeepSeek-R1", "mistral-ai/Mistral-Large-2411"],
        "default":    "openai/gpt-4o",
        "quota":      "Gratuit · 15 RPM · 150 req/jour · Fine-grained PAT",
        "vision":     True,
        "type":       "github_models",
        "base_url":   "https://models.github.ai/inference",
        "note":       "Token Fine-grained PAT avec permission models:read requis",
    },
}


def get_active_provider():
    return get_setting("ia_provider", list(IA_PROVIDERS.keys())[0])


def get_active_model():
    provider = get_active_provider()
    saved = get_setting("ia_model", "")
    if saved and saved in IA_PROVIDERS.get(provider, {}).get("models", []):
        return saved
    return IA_PROVIDERS.get(provider, {}).get("default", "")


def get_api_key_for_provider(provider_name):
    cfg = IA_PROVIDERS.get(provider_name, {})
    key = get_setting(cfg.get("key", ""), "")
    if not key:
        key = os.environ.get(cfg.get("env", ""), "")
    return key


def ia_call(prompt_text, image_bytes=None, json_mode=False):
    """
    Appel unifié vers le fournisseur IA actif.
    Supporte : Anthropic, Google, Groq, OpenRouter, Mistral, Cohere,
               Zhipu, Cerebras, HuggingFace, GitHub Models.
    """
    import urllib.error

    provider_name = get_active_provider()
    model         = get_active_model()
    api_key       = get_api_key_for_provider(provider_name)
    cfg           = IA_PROVIDERS.get(provider_name, {})
    ptype         = cfg.get("type", "")

    if not api_key:
        return None

    try:
        # ── 1. ANTHROPIC ──────────────────────────────────────────────────
        if ptype == "anthropic" and ANTHROPIC_OK:
            client = anthropic.Anthropic(api_key=api_key)
            content = []
            if image_bytes and cfg.get("vision"):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": base64.b64encode(image_bytes).decode()}
                })
            content.append({"type": "text", "text": prompt_text})
            resp = client.messages.create(model=model, max_tokens=2000,
                                          messages=[{"role": "user", "content": content}])
            return resp.content[0].text

        # ── 2. GOOGLE (Gemini API) ────────────────────────────────────────
        elif ptype == "google":
            import urllib.request
            parts = []
            if image_bytes and cfg.get("vision"):
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode()
                    }
                })
            parts.append({"text": prompt_text})
            payload = json.dumps({"contents": [{"parts": parts}]}).encode()
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={api_key}")
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]

        # ── 3. COHERE v2 ──────────────────────────────────────────────────
        elif ptype == "cohere":
            import urllib.request
            body = {
                "model":    model,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 2000,
                "temperature": 0.3,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            payload = json.dumps(body).encode()
            req = urllib.request.Request(
                "https://api.cohere.com/v2/chat",
                data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Accept":        "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            msg = data.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list) and content:
                return content[0].get("text", str(content))
            return str(content)

        # ── 4. HUGGING FACE ───────────────────────────────────────────────
        elif ptype == "huggingface":
            import urllib.request
            body = {
                "model":    model,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 1800,
                "temperature": 0.4,
                "stream": False,
            }
            payload = json.dumps(body).encode()
            url = "https://api-inference.huggingface.co/v1/chat/completions"
            req = urllib.request.Request(
                url, data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            if isinstance(data, list):
                full = data[0].get("generated_text", "")
                if full.startswith(prompt_text):
                    return full[len(prompt_text):].strip()
                return full
            return str(data)

        # ── 5. OPENAI-COMPATIBLE (Groq, OpenRouter, Mistral, Cerebras, Zhipu) ──
        elif ptype == "openai_compat":
            import urllib.request
            base_url = cfg.get("base_url", "")
            messages = []
            if image_bytes and cfg.get("vision"):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,"
                                              f"{base64.b64encode(image_bytes).decode()}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt_text})
            body = {"model": model, "messages": messages,
                    "max_tokens": 2000, "temperature": 0.3}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            payload = json.dumps(body).encode()
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {api_key}"}
            if "openrouter" in base_url:
                headers["HTTP-Referer"] = "https://apitrack.pro"
                headers["X-Title"] = "ApiTrack Pro"
            req = urllib.request.Request(f"{base_url}/chat/completions",
                                         data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]

        # ── 6. GITHUB MODELS ──────────────────────────────────────────────
        elif ptype == "github_models":
            import urllib.request
            endpoint = "https://models.github.ai/inference/chat/completions"
            messages = []
            if image_bytes and cfg.get("vision"):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
                         }},
                        {"type": "text", "text": prompt_text}
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt_text})
            body = {
                "model":       model,
                "messages":    messages,
                "max_tokens":  2000,
                "temperature": 0.3,
            }
            if json_mode and model.startswith("openai/"):
                body["response_format"] = {"type": "json_object"}
            payload = json.dumps(body).encode()
            headers = {
                "Content-Type":         "application/json",
                "Accept":               "application/vnd.github+json",
                "Authorization":        f"Bearer {api_key}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            req = urllib.request.Request(endpoint, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]

        return None

    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:400]
        except Exception:
            pass
        if e.code == 401:
            if ptype == "github_models":
                return (f"❌ GitHub Models — Authentification échouée (401).\n"
                        f"→ Utilisez un Fine-grained PAT (github_pat_...)\n"
                        f"→ Permission requise : Models → Read-only")
            return f"❌ Erreur {provider_name} : HTTP 401 — vérifiez votre clé API. {body}"
        elif e.code == 404:
            return f"❌ Erreur {provider_name} : HTTP 404 — endpoint ou modèle introuvable. {body}"
        elif e.code == 429:
            return f"❌ Erreur {provider_name} : Quota dépassé (429) — attendez quelques minutes. {body}"
        elif e.code == 422:
            return f"❌ Erreur {provider_name} : Paramètres invalides (422). {body}"
        else:
            return f"❌ Erreur {provider_name} : HTTP {e.code} {e.reason}. {body}"
    except Exception as e:
        return f"❌ Erreur {provider_name} : {e}"


def ia_call_json(prompt_text, image_bytes=None):
    """Appel IA avec retour JSON parsé."""
    result = ia_call(prompt_text, image_bytes, json_mode=True)
    if not result or result.startswith("❌"):
        return {"error": result or "Pas de réponse"}
    text = result.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if p.startswith("json"):
                text = p[4:].strip()
                break
            elif p.strip().startswith("{"):
                text = p.strip()
                break
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"error": f"JSON invalide : {text[:200]}"}


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS IA MÉTIER — utilisent ia_call() → tous fournisseurs supportés
# ════════════════════════════════════════════════════════════════════════════

def ia_analyser_morphometrie(aile, largeur, cubital, glossa, tomentum, pigmentation,
                              race_algo, confiance, image_bytes=None):
    """
    Analyse morphométrique via le fournisseur IA ACTIF (Gemma, Claude, Groq, etc.)
    Plus de dépendance forcée à Anthropic.
    """
    pname = get_active_provider()
    model = get_active_model()
    prompt = f"""Tu es expert apicole et morphométriste spécialisé dans la classification des races d'abeilles selon Ruttner (1988).

Voici les mesures morphométriques relevées sur une abeille :
- Longueur aile antérieure : {aile} mm
- Largeur aile : {largeur} mm
- Indice cubital : {cubital}
- Longueur glossa : {glossa} mm
- Tomentum (densité poils thorax 0-3) : {tomentum}
- Pigmentation scutellum : {pigmentation}

L'algorithme local a classifié : **{race_algo}** avec {confiance}% de confiance.
Modèle IA utilisé : {pname} / {model}

Effectue une analyse morphométrique complète en français selon ce plan :

## 1. Validation de la classification
- Confirme ou nuance la race {race_algo} selon les valeurs Ruttner 1988
- Ton niveau de confiance personnel (0-100%)
- Comparaison avec A.m. intermissa, sahariensis, ligustica, carnica

## 2. Scores de production (note /5 ⭐)
- 🍯 **Miel** : X/5 — justification (rendement kg/ruche/an estimé)
- 🌼 **Pollen** : X/5 — justification
- 🟤 **Propolis** : X/5 — justification
- 👑 **Gelée royale** : X/5 — justification (taux 10-HDA estimé)

## 3. Caractéristiques comportementales
Douceur, essaimage, économie hivernale, résistance varroa (2-3 lignes)

## 4. Recommandations stratégiques (3 actions concrètes)
- Action 1 :
- Action 2 :
- Action 3 :

## 5. Compatibilité avec l'environnement nord-africain (Algérie/Maroc/Tunisie)
Court paragraphe sur l'adaptation de cette race au climat méditerranéen/saharien.

Sois précis, concis, vocabulaire apicole professionnel."""
    return ia_call(prompt, image_bytes)


def ia_analyser_environnement(description_env, latitude=None, longitude=None,
                               saison="printemps", image_bytes=None):
    """
    Analyse environnementale mellifère via le fournisseur IA ACTIF.
    Fonctionne avec Gemma, Claude, Groq, Mistral, etc.
    """
    pname = get_active_provider()
    coords_str = f"Coordonnées : {latitude:.4f}°N, {longitude:.4f}°E" if latitude else ""
    prompt = f"""Tu es expert apicole senior, botaniste et écologue spécialisé dans l'analyse des environnements mellifères méditerranéens et nord-africains.

Zone à analyser :
{coords_str}
Saison : {saison}
Description : {description_env}
IA utilisée : {pname}

Effectue une analyse environnementale mellifère COMPLÈTE en français :

## 🌿 1. Flore identifiée et potentiel mellifère
Pour chaque espèce présente ou probable :
| Espèce | Source | Période | Qualité |
(Nectar / Pollen / Résine / Mixte — Excellente/Bonne/Moyenne/Faible)

## 📊 2. Scores de production (note /5 ⭐)
- 🍯 **MIEL** : X/5 — (type floral, saveur probable, rendement estimé kg/ruche/an, période)
- 🌼 **POLLEN** : X/5 — (diversité, richesse protéique %, couleurs)
- 🟤 **PROPOLIS** : X/5 — (espèces résineuses, qualité antibactérienne estimée)
- 👑 **GELÉE ROYALE** : X/5 — (disponibilité protéines+sucres, taux 10-HDA estimé)

## 🌡️ 3. Analyse microclimatique
- Exposition, altitude, humidité, vent, eau permanente
- Risques : pesticides, sécheresse, concurrence, prédateurs
- Points forts spécifiques à cette zone

## 🎯 4. Verdict global
- Potentiel global : [Faible/Modéré/Élevé/Exceptionnel]
- Indice mellifère : X/10
- Production principale recommandée : [Miel/Pollen/Propolis/Gelée royale/Mixte]
- Capacité de charge : X ruches/100 ha

## 🐝 5. Plan d'action (5 recommandations)
- Race d'abeille la plus adaptée à cette zone
- Mois optimal d'installation des ruches
- Période de récolte recommandée
- 3 améliorations pour maximiser la production

Données chiffrées obligatoires. Références botaniques locales nord-africaines si possible."""
    return ia_call(prompt, image_bytes)


def ia_analyser_zone_carto(nom_zone, flore, superficie, ndvi, potentiel, type_zone,
                            latitude=None, longitude=None):
    """
    Analyse JSON d'une zone cartographiée via le fournisseur IA ACTIF.
    Fonctionne avec Gemma, Claude, Groq, Mistral, etc.
    """
    coords_str = f"à {latitude:.4f}°N, {longitude:.4f}°E" if latitude else ""
    prompt = f"""Tu es expert apicole et écologue. Analyse cette zone mellifère cartographiée.

Zone : {nom_zone} {coords_str}
Type : {type_zone} | Flore : {flore} | Superficie : {superficie} ha
NDVI : {ndvi} (0=sol nu → 1=végétation dense) | Potentiel estimé : {potentiel}

Réponds UNIQUEMENT avec un objet JSON valide (pas de texte avant/après, pas de markdown) :
{{
  "diagnostic": {{"potentiel_global":"Élevé","indice_mellifere":8,"capacite_ruches":12,"saison_pic":"Avril-Juin"}},
  "scores": {{
    "miel":{{"note":4,"etoiles":"⭐⭐⭐⭐","detail":"Nectar abondant — jujubier dominant"}},
    "pollen":{{"note":3,"etoiles":"⭐⭐⭐","detail":"Diversité florale correcte"}},
    "propolis":{{"note":2,"etoiles":"⭐⭐","detail":"Quelques résines disponibles"}},
    "gelee_royale":{{"note":3,"etoiles":"⭐⭐⭐","detail":"Protéines disponibles printemps"}}
  }},
  "flore_identifiee":[
    {{"espece":"Ziziphus lotus","nectar":true,"pollen":true,"resine":false,"periode":"Avr-Juin","qualite":"Excellente"}}
  ],
  "risques":["Sécheresse estivale","Faible diversité florale en été"],
  "recommandations":["Installer 8-12 ruches en mars","Récolter miel en juin","Prévoir nourrissement été"],
  "race_adaptee":"intermissa",
  "resume":"Zone mellifère de haute valeur — potentiel miel jujubier exceptionnel au printemps."
}}"""
    return ia_call_json(prompt)


def afficher_resultat_ia(texte, titre="🤖 Analyse IA"):
    """Affiche le résultat IA dans un bloc stylisé avec badge fournisseur."""
    provider = get_active_provider()
    model    = get_active_model()
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#161B27,#1E2535);
                border:1px solid #C8820A;border-left:4px solid #C8820A;
                border-radius:10px;padding:20px;margin:16px 0;'>
        <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:12px'>
            <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:600;color:#F5A623'>
                🤖 {titre}
            </div>
            <div style='font-size:.7rem;background:#1E2010;color:#A8B4CC;border:1px solid #2E3A52;
                        border-radius:20px;padding:2px 10px'>{provider} · {model}</div>
        </div>
        <div style='font-size:.88rem;color:#F0F4FF;line-height:1.7'>
    """, unsafe_allow_html=True)
    st.markdown(texte)
    st.markdown("</div></div>", unsafe_allow_html=True)


# Alias de compatibilité
def afficher_resultat_ia_zone(texte, titre="🤖 Analyse IA"):
    afficher_resultat_ia(texte, titre)


def widget_ia_selector():
    """
    Widget sélecteur de fournisseur IA.
    Retourne True si une clé est configurée pour le fournisseur actif.
    """
    provider_names = list(IA_PROVIDERS.keys())
    current = get_active_provider()
    idx = provider_names.index(current) if current in provider_names else 0

    with st.expander("🤖 Choisir le fournisseur IA", expanded=False):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            sel = st.selectbox("Fournisseur IA gratuit", provider_names,
                                index=idx, key="ia_provider_select")
        cfg = IA_PROVIDERS[sel]
        with col2:
            models = cfg["models"]
            current_model = get_setting("ia_model", cfg["default"])
            idx_m = models.index(current_model) if current_model in models else 0
            sel_model = st.selectbox("Modèle", models, index=idx_m, key="ia_model_select")

        st.markdown(f"""
        <div style='font-size:.78rem;color:#A8B4CC;background:#0F1117;border-radius:6px;
                    padding:8px 12px;margin:6px 0;line-height:1.6'>
        📊 <b>Quota :</b> {cfg['quota']}<br>
        🖼️ <b>Vision (photo) :</b> {'✅ Oui' if cfg['vision'] else '❌ Texte seul'}<br>
        🔑 <b>Obtenir la clé :</b> <a href='{cfg['url']}' target='_blank'>{cfg['url']}</a>
        {f"<br>⚠️ <b>Note :</b> {cfg['note']}" if cfg.get('note') else ""}
        </div>
        """, unsafe_allow_html=True)

        if cfg.get("type") == "github_models":
            st.markdown("""
            <div style='background:#0D1A2A;border:1px solid #1A3A5C;border-radius:6px;
                        padding:10px 14px;font-size:.78rem;color:#F0F4FF;margin-bottom:8px'>
            <b>🐙 Comment créer le bon token GitHub :</b><br>
            1. Allez sur <a href='https://github.com/settings/personal-access-tokens/new' target='_blank'>
               github.com/settings/personal-access-tokens/new</a><br>
            2. Choisissez <b>"Fine-grained personal access token"</b><br>
            3. Dans <b>Permissions → Account permissions</b> → <b>Models</b> → <b>Read-only</b><br>
            4. Cliquez <b>Generate token</b> → copiez le token (<code>github_pat_...</code>)<br>
            5. <b>⚠️ Les tokens classiques <code>ghp_...</code> ne fonctionnent PAS</b>
            </div>
            """, unsafe_allow_html=True)

        api_key = get_api_key_for_provider(sel)
        new_key = st.text_input(
            f"Clé API {sel.split('(')[0].strip()}",
            value=api_key, type="password",
            placeholder=cfg.get("prefix", "") + "...",
            key=f"key_input_{sel}"
        )

        col_s1, col_s2 = st.columns(2)
        if col_s1.button("💾 Sauvegarder & Activer", key="save_ia_provider"):
            conn = get_db()
            if new_key:
                conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)",
                             (cfg["key"], new_key))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('ia_provider',?)", (sel,))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('ia_model',?)", (sel_model,))
            conn.commit()
            conn.close()
            log_action("Fournisseur IA changé", f"{sel} / {sel_model}")
            st.success(f"✅ {sel} activé — modèle {sel_model}")
            st.rerun()
        if col_s2.button("🔬 Tester la connexion", key="test_ia_provider"):
            conn = get_db()
            if new_key:
                conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (cfg["key"], new_key))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('ia_provider',?)", (sel,))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('ia_model',?)", (sel_model,))
            conn.commit()
            conn.close()
            with st.spinner("Test en cours..."):
                r = ia_call("Réponds uniquement : 'ApiTrack Pro IA OK' en français.")
            if r and "OK" in r:
                st.success(f"✅ {r.strip()}")
            elif r:
                st.warning(f"Réponse : {r[:200]}")
            else:
                st.error("❌ Pas de réponse. Vérifiez la clé API.")

    api_key = get_api_key_for_provider(get_active_provider())
    prov    = get_active_provider()
    mod     = get_active_model()
    if api_key:
        st.markdown(f"<div style='font-size:.75rem;color:#6EE7B7;margin-bottom:8px'>"
                    f"✅ IA active : <b>{prov}</b> · <code>{mod}</code></div>",
                    unsafe_allow_html=True)
        return True
    else:
        st.warning(f"⚠️ Configurez une clé API pour **{prov}** (voir le sélecteur ci-dessus).")
        return False


# Alias de compatibilité
def widget_cle_api():
    return widget_ia_selector()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:8px 0 16px;border-bottom:1px solid #3d2a0e;margin-bottom:12px'>
            <div style='font-size:1.6rem;margin-bottom:4px'>🐝</div>
            <div style='font-family:Playfair Display,serif;color:#F5A623;font-size:1.1rem;font-weight:600'>ApiTrack Pro</div>
            <div style='font-size:.65rem;color:#8899BB;text-transform:uppercase;letter-spacing:.1em'>Gestion Apicole</div>
        </div>
        """, unsafe_allow_html=True)

        rucher_nom = get_setting("rucher_nom", "Mon Rucher")
        st.markdown(f"<div style='font-size:.75rem;color:#6B7A99;margin-bottom:12px'>📍 {rucher_nom}</div>", unsafe_allow_html=True)

        pages = {
            "🏠 Dashboard": "dashboard",
            "🐝 Mes ruches": "ruches",
            "🔍 Inspections": "inspections",
            "💊 Traitements": "traitements",
            "🍯 Productions": "productions",
            "🧬 Morphométrie IA": "morpho",
            "🗺️ Cartographie": "carto",
            "☀️ Météo & Miellée": "meteo",
            "📊 Génétique": "genetique",
            "🌿 Flore mellifère": "flore",
            "⚠️ Alertes": "alertes",
            "📋 Journal": "journal",
            "⚙️ Administration": "admin",
        }

        if "page" not in st.session_state:
            st.session_state.page = "dashboard"

        for label, key in pages.items():
            if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.sidebar.markdown("<hr style='border-color:#2E3A52;margin:12px 0'>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size:.75rem;color:#6B7A99'>👤 {st.session_state.get('username','admin')}</div>", unsafe_allow_html=True)
        if st.sidebar.button("🚪 Déconnexion", use_container_width=True):
            log_action("Déconnexion", f"Utilisateur {st.session_state.get('username')} déconnecté")
            st.session_state.logged_in = False
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("## 🏠 Tableau de bord")
    rucher = get_setting("rucher_nom", "Mon Rucher")
    localisation = get_setting("localisation", "")
    st.markdown(f"<p style='color:#A8B4CC;margin-top:-10px'>Saison printanière 2025 · {rucher} · {localisation}</p>", unsafe_allow_html=True)

    conn = get_db()
    nb_ruches = conn.execute("SELECT COUNT(*) FROM ruches WHERE statut='actif'").fetchone()[0]
    total_miel = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='miel'").fetchone()[0]
    nb_insp = conn.execute("SELECT COUNT(*) FROM inspections WHERE date_inspection >= date('now','-30 days')").fetchone()[0]
    critiques = conn.execute("SELECT COUNT(*) FROM inspections WHERE varroa_pct >= 3.0 AND date_inspection >= date('now','-7 days')").fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🐝 Ruches actives", nb_ruches, "+3 ce mois")
    col2.metric("🍯 Miel récolté (kg)", f"{total_miel:.0f}", "+18% vs 2024")
    col3.metric("🔍 Inspections (30j)", nb_insp, "Cadence correcte")
    col4.metric("⚠️ Varroa critique", critiques, "Intervention requise" if critiques else "RAS", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📈 Production mensuelle (kg)")
        df_prod = pd.read_sql("""
            SELECT strftime('%Y-%m', date_recolte) as mois,
                   type_produit,
                   SUM(quantite_kg) as total
            FROM recoltes
            GROUP BY mois, type_produit
            ORDER BY mois
        """, conn)
        if not df_prod.empty:
            fig = px.bar(df_prod, x="mois", y="total", color="type_produit",
                         color_discrete_map={"miel":"#C8820A","pollen":"#F5C842","gelée royale":"#8B7355"},
                         template="plotly_white")
            fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              legend_title_text="", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de production.")

    with col_b:
        st.markdown("### 🐝 État des ruches")
        df_ruches = pd.read_sql("""
            SELECT r.nom, r.race,
                   COALESCE(i.varroa_pct, 0) as varroa,
                   COALESCE(i.nb_cadres, 0) as cadres,
                   COALESCE(i.poids_kg, 0) as poids
            FROM ruches r
            LEFT JOIN inspections i ON i.ruche_id = r.id
            AND i.date_inspection = (SELECT MAX(ii.date_inspection) FROM inspections ii WHERE ii.ruche_id = r.id)
            WHERE r.statut='actif'
            ORDER BY varroa DESC
            LIMIT 6
        """, conn)
        if not df_ruches.empty:
            df_ruches["Statut"] = df_ruches["varroa"].apply(status_badge)
            df_ruches.columns = ["Ruche","Race","Varroa%","Cadres","Poids(kg)","Statut"]
            st.dataframe(df_ruches, use_container_width=True, hide_index=True)

    st.markdown("### ⚠️ Alertes actives")
    df_alertes = pd.read_sql("""
        SELECT r.nom, i.varroa_pct, i.date_inspection, i.notes
        FROM inspections i
        JOIN ruches r ON r.id = i.ruche_id
        WHERE i.varroa_pct >= 2.0
        AND i.date_inspection >= date('now','-7 days')
        ORDER BY i.varroa_pct DESC
    """, conn)
    conn.close()

    if not df_alertes.empty:
        for _, row in df_alertes.iterrows():
            lvl = "🔴" if row["varroa_pct"] >= 3.0 else "🟡"
            seuil = "CRITIQUE (>3%)" if row["varroa_pct"] >= 3.0 else "ATTENTION (>2%)"
            st.warning(f"{lvl} **{row['nom']}** — Varroa **{row['varroa_pct']}%** — {seuil} · {row['date_inspection']}")
    else:
        st.success("✅ Aucune alerte varroa critique en cours.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE : GESTION DES RUCHES
# ════════════════════════════════════════════════════════════════════════════
def page_ruches():
    st.markdown("## 🐝 Gestion des ruches")

    conn = get_db()
    df = pd.read_sql("""
        SELECT r.id, r.nom, r.race, r.date_installation, r.localisation, r.statut,
               COALESCE(i.varroa_pct, '-') as derniere_varroa,
               COALESCE(i.nb_cadres, '-') as cadres,
               COALESCE(i.poids_kg, '-') as poids_kg,
               i.date_inspection as derniere_inspection
        FROM ruches r
        LEFT JOIN inspections i ON i.ruche_id = r.id
        AND i.date_inspection = (SELECT MAX(ii.date_inspection) FROM inspections ii WHERE ii.ruche_id = r.id)
        ORDER BY r.id
    """, conn)

    tab1, tab2 = st.tabs(["📋 Liste des ruches", "➕ Ajouter une ruche"])

    with tab1:
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv, "ruches.csv", "text/csv")

        st.markdown("### 🗑️ Supprimer une ruche")
        ruche_ids = conn.execute("SELECT id, nom FROM ruches").fetchall()
        if ruche_ids:
            options = {f"R{r[0]:02d} — {r[1]}": r[0] for r in ruche_ids}
            selected = st.selectbox("Choisir la ruche à supprimer", options.keys())
            if st.button("⚠️ Supprimer définitivement", type="secondary"):
                rid = options[selected]
                conn.execute("DELETE FROM ruches WHERE id=?", (rid,))
                conn.commit()
                log_action("Suppression ruche", f"Ruche {selected} supprimée")
                st.success(f"Ruche {selected} supprimée.")
                st.rerun()

    with tab2:
        with st.form("add_ruche"):
            st.markdown("**Nouvelle ruche**")
            col1, col2 = st.columns(2)
            nom = col1.text_input("Nom / Reine*")
            race = col2.selectbox("Race", ["intermissa", "sahariensis", "ligustica", "carnica", "hybride"])
            date_inst = col1.date_input("Date d'installation", datetime.date.today())
            localisation = col2.text_input("Localisation")
            col3, col4 = st.columns(2)
            lat = col3.number_input("Latitude", value=34.88, format="%.4f")
            lon = col4.number_input("Longitude", value=1.32, format="%.4f")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("✅ Ajouter la ruche")

        if submitted and nom:
            conn.execute("""
                INSERT INTO ruches (nom, race, date_installation, localisation, latitude, longitude, notes)
                VALUES (?,?,?,?,?,?,?)
            """, (nom, race, str(date_inst), localisation, lat, lon, notes))
            conn.commit()
            log_action("Ajout ruche", f"Ruche '{nom}' ({race}) ajoutée")
            st.success(f"✅ Ruche '{nom}' ajoutée avec succès.")
            st.rerun()

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : INSPECTIONS
# ════════════════════════════════════════════════════════════════════════════
def page_inspections():
    st.markdown("## 🔍 Inspections")
    conn = get_db()

    tab1, tab2 = st.tabs(["📋 Historique", "➕ Nouvelle inspection"])

    with tab1:
        df = pd.read_sql("""
            SELECT i.id, r.nom as ruche, i.date_inspection, i.poids_kg, i.nb_cadres,
                   i.varroa_pct, i.reine_vue, i.comportement, i.notes
            FROM inspections i
            JOIN ruches r ON r.id = i.ruche_id
            ORDER BY i.date_inspection DESC
        """, conn)
        if not df.empty:
            df["reine_vue"] = df["reine_vue"].apply(lambda x: "✓" if x else "✗")
            df["varroa_pct"] = df["varroa_pct"].apply(lambda x: f"{x}%" if x else "-")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv, "inspections.csv", "text/csv")

        st.markdown("### 📈 Évolution du varroa")
        df_v = pd.read_sql("""
            SELECT r.nom, i.date_inspection, i.varroa_pct
            FROM inspections i JOIN ruches r ON r.id=i.ruche_id
            WHERE i.varroa_pct IS NOT NULL
            ORDER BY i.date_inspection
        """, conn)
        if not df_v.empty:
            fig = px.line(df_v, x="date_inspection", y="varroa_pct", color="nom",
                          template="plotly_white", markers=True)
            fig.add_hline(y=2.0, line_dash="dash", line_color="orange", annotation_text="Seuil alerte (2%)")
            fig.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="Seuil critique (3%)")
            fig.update_layout(height=300, margin=dict(t=10,b=10,l=0,r=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
        opts = {r[1]: r[0] for r in ruches}

        with st.form("add_inspection"):
            col1, col2 = st.columns(2)
            ruche_sel = col1.selectbox("Ruche*", opts.keys())
            date_insp = col2.date_input("Date", datetime.date.today())
            col3, col4, col5 = st.columns(3)
            poids = col3.number_input("Poids (kg)", 0.0, 80.0, 25.0, 0.1)
            cadres = col4.number_input("Nb cadres", 0, 20, 10)
            varroa = col5.number_input("Varroa (%)", 0.0, 20.0, 1.0, 0.1)
            col6, col7 = st.columns(2)
            reine = col6.checkbox("Reine vue", value=True)
            comportement = col7.selectbox("Comportement", ["calme", "nerveuse", "agressive", "très calme"])
            notes = st.text_area("Notes / Observations")
            submitted = st.form_submit_button("✅ Enregistrer l'inspection")

        if submitted:
            rid = opts[ruche_sel]
            conn.execute("""
                INSERT INTO inspections (ruche_id,date_inspection,poids_kg,nb_cadres,varroa_pct,reine_vue,comportement,notes)
                VALUES (?,?,?,?,?,?,?,?)
            """, (rid, str(date_insp), poids, cadres, varroa, int(reine), comportement, notes))
            conn.commit()
            log_action("Inspection enregistrée", f"Ruche {ruche_sel} — varroa {varroa}%")
            if varroa >= 3.0:
                st.error(f"⚠️ ALERTE CRITIQUE : Varroa {varroa}% sur {ruche_sel} — Traitement immédiat requis !")
            elif varroa >= 2.0:
                st.warning(f"⚠️ Attention : Varroa {varroa}% sur {ruche_sel} — Surveillance renforcée.")
            else:
                st.success("✅ Inspection enregistrée.")
            st.rerun()

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : TRAITEMENTS
# ════════════════════════════════════════════════════════════════════════════
def page_traitements():
    st.markdown("## 💊 Traitements vétérinaires")
    conn = get_db()

    tab1, tab2 = st.tabs(["📋 En cours & historique", "➕ Nouveau traitement"])

    with tab1:
        df = pd.read_sql("""
            SELECT t.id, r.nom as ruche, t.date_debut, t.date_fin, t.produit,
                   t.pathologie, t.dose, t.duree_jours, t.statut, t.notes
            FROM traitements t JOIN ruches r ON r.id=t.ruche_id
            ORDER BY t.date_debut DESC
        """, conn)
        if not df.empty:
            for _, row in df.iterrows():
                if row["statut"] == "en_cours":
                    debut = datetime.date.fromisoformat(row["date_debut"])
                    jours_ecoulés = (datetime.date.today() - debut).days
                    duree = row["duree_jours"] or 21
                    progress = min(jours_ecoulés / duree, 1.0)
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        col1.markdown(f"**{row['ruche']}** — {row['produit']} ({row['pathologie']}) · Dose : {row['dose']}")
                        col1.progress(progress, text=f"Jour {jours_ecoulés}/{duree}")
                        col2.markdown(f"<span class='badge-warn'>En cours</span>", unsafe_allow_html=True)
                    st.markdown("---")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun traitement enregistré.")

    with tab2:
        ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
        opts = {r[1]: r[0] for r in ruches}
        with st.form("add_traitement"):
            col1, col2 = st.columns(2)
            ruche_sel = col1.selectbox("Ruche", opts.keys())
            produit = col2.text_input("Produit", placeholder="Acide oxalique")
            col3, col4 = st.columns(2)
            pathologie = col3.selectbox("Pathologie", ["Varroa", "Loque américaine", "Nosémose", "Foulbrood", "Autre"])
            dose = col4.text_input("Dose", placeholder="50 ml")
            col5, col6 = st.columns(2)
            date_debut = col5.date_input("Date début", datetime.date.today())
            duree = col6.number_input("Durée (jours)", 1, 90, 21)
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("✅ Enregistrer le traitement")

        if submitted and produit:
            date_fin = date_debut + datetime.timedelta(days=duree)
            conn.execute("""
                INSERT INTO traitements (ruche_id,date_debut,date_fin,produit,pathologie,dose,duree_jours,statut,notes)
                VALUES (?,?,?,?,?,?,?,'en_cours',?)
            """, (opts[ruche_sel], str(date_debut), str(date_fin), produit, pathologie, dose, duree, notes))
            conn.commit()
            log_action("Traitement débuté", f"Ruche {ruche_sel} — {produit} ({pathologie})")
            st.success("✅ Traitement enregistré.")
            st.rerun()

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : PRODUCTIONS
# ════════════════════════════════════════════════════════════════════════════
def page_productions():
    st.markdown("## 🍯 Productions")
    conn = get_db()

    total_miel = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='miel'").fetchone()[0]
    total_pollen = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='pollen'").fetchone()[0]
    total_gr = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='gelée royale'").fetchone()[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🍯 Miel total (kg)", f"{total_miel:.1f}", "Humidité moy. 17.2%")
    col2.metric("🌼 Pollen (kg)", f"{total_pollen:.1f}", "Qualité A")
    col3.metric("👑 Gelée royale (kg)", f"{total_gr:.2f}", "10-HDA 2.1%")

    tab1, tab2, tab3 = st.tabs(["🍯 Récoltes", "📊 Graphiques", "➕ Nouvelle récolte"])

    with tab1:
        df = pd.read_sql("""
            SELECT rec.id, r.nom as ruche, rec.date_recolte, rec.type_produit,
                   rec.quantite_kg, rec.humidite_pct, rec.ph, rec.hda_pct, rec.qualite, rec.notes
            FROM recoltes rec JOIN ruches r ON r.id=rec.ruche_id
            ORDER BY rec.date_recolte DESC
        """, conn)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv, "recoltes.csv", "text/csv")

    with tab2:
        df_g = pd.read_sql("""
            SELECT strftime('%Y-%m', date_recolte) as mois, type_produit, SUM(quantite_kg) as total
            FROM recoltes GROUP BY mois, type_produit ORDER BY mois
        """, conn)
        if not df_g.empty:
            fig = px.area(df_g, x="mois", y="total", color="type_produit",
                          color_discrete_map={"miel":"#C8820A","pollen":"#F5C842","gelée royale":"#8B7355"},
                          template="plotly_white")
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10,b=10,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

        df_r = pd.read_sql("""
            SELECT r.nom, SUM(rec.quantite_kg) as total FROM recoltes rec
            JOIN ruches r ON r.id=rec.ruche_id WHERE rec.type_produit='miel'
            GROUP BY r.nom ORDER BY total DESC
        """, conn)
        if not df_r.empty:
            fig2 = px.bar(df_r, x="nom", y="total", template="plotly_white",
                          color_discrete_sequence=["#C8820A"])
            fig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=10,b=10,l=0,r=0), title="Production de miel par ruche (kg)")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
        opts = {r[1]: r[0] for r in ruches}
        with st.form("add_recolte"):
            col1, col2, col3 = st.columns(3)
            ruche_sel = col1.selectbox("Ruche", opts.keys())
            type_prod = col2.selectbox("Produit", ["miel", "pollen", "gelée royale", "propolis"])
            date_rec = col3.date_input("Date récolte", datetime.date.today())
            col4, col5 = st.columns(2)
            quantite = col4.number_input("Quantité (kg)", 0.0, 500.0, 10.0, 0.1)
            qualite = col5.selectbox("Qualité", ["A+", "A", "B", "C"])
            col6, col7, col8 = st.columns(3)
            humidite = col6.number_input("Humidité (%)", 0.0, 30.0, 17.5, 0.1)
            ph = col7.number_input("pH", 2.0, 7.0, 3.9, 0.1)
            hda = col8.number_input("10-HDA (%)", 0.0, 5.0, 0.0, 0.1)
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("✅ Enregistrer la récolte")

        if submitted:
            conn.execute("""
                INSERT INTO recoltes (ruche_id,date_recolte,type_produit,quantite_kg,humidite_pct,ph,hda_pct,qualite,notes)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (opts[ruche_sel], str(date_rec), type_prod, quantite,
                  humidite if humidite > 0 else None,
                  ph if ph > 0 else None,
                  hda if hda > 0 else None, qualite, notes))
            conn.commit()
            log_action("Récolte enregistrée", f"{quantite} kg de {type_prod} — ruche {ruche_sel}")
            st.success(f"✅ {quantite} kg de {type_prod} enregistrés.")
            st.rerun()

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : MORPHOMÉTRIE IA
# ════════════════════════════════════════════════════════════════════════════
RUTTNER_REF = {
    "intermissa":   {"aile": (8.9, 9.4), "cubital": (2.0, 2.8), "glossa": (5.8, 6.3)},
    "sahariensis":  {"aile": (9.0, 9.5), "cubital": (1.9, 2.5), "glossa": (6.0, 6.5)},
    "ligustica":    {"aile": (9.2, 9.8), "cubital": (2.5, 3.2), "glossa": (6.3, 6.8)},
    "carnica":      {"aile": (9.3, 9.9), "cubital": (2.2, 3.0), "glossa": (6.4, 7.0)},
    "hybride":      {"aile": (8.5, 9.5), "cubital": (1.8, 3.5), "glossa": (5.5, 6.8)},
}

def classify_race(aile, cubital, glossa):
    scores = {}
    for race, ref in RUTTNER_REF.items():
        s = 0
        for val, (lo, hi) in [(aile, ref["aile"]), (cubital, ref["cubital"]), (glossa, ref["glossa"])]:
            if val is None:
                s += 0.5
            elif lo <= val <= hi:
                s += 1.0
            else:
                dist = min(abs(val - lo), abs(val - hi))
                s += max(0, 1.0 - dist * 0.5)
        scores[race] = s
    total = sum(scores.values()) or 1
    return {r: round(v / total * 100) for r, v in scores.items()}


def page_morpho():
    st.markdown("## 🧬 Morphométrie IA — Classification raciale")
    st.markdown("<p style='color:#A8B4CC'>Mesures morphométriques + analyse IA multi-fournisseurs (Ruttner 1988)</p>",
                unsafe_allow_html=True)

    ia_active = widget_cle_api()

    conn = get_db()
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    opts = {r[1]: r[0] for r in ruches}

    specialisations = {
        "intermissa": ["Production de miel", "Propolis abondante", "Résistance chaleur", "Adaptation locale"],
        "sahariensis": ["Butinage intense", "Résistance extrême chaleur", "Économie eau"],
        "ligustica": ["Production intensive miel", "Faible propolis", "Docilité"],
        "carnica": ["Économie hivernale", "Butinage précoce", "Faible essaimage"],
        "hybride": ["Variable selon parentaux", "Évaluation approfondie requise"],
    }

    tab1, tab2 = st.tabs(["🔬 Analyse + IA", "📜 Historique"])

    with tab1:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### 📐 Mesures morphométriques")
            ruche_sel = st.selectbox("Ruche analysée", opts.keys())
            aile    = st.number_input("Longueur aile antérieure (mm)", 7.0, 12.0, 9.2, 0.1)
            largeur = st.number_input("Largeur aile (mm)", 2.0, 5.0, 3.1, 0.1)
            cubital = st.number_input("Indice cubital", 1.0, 5.0, 2.3, 0.1,
                                      help="Rapport distances nervures cubitales a/b ÷ b/c")
            glossa  = st.number_input("Longueur glossa (mm)", 4.0, 8.0, 6.1, 0.1)
            tomentum    = st.slider("Tomentum (densité poils thorax 0–3)", 0, 3, 2)
            pigmentation = st.selectbox("Pigmentation scutellum",
                                        ["Noir", "Brun foncé", "Brun clair", "Jaune"])
            notes = st.text_area("Notes / Observations")

            st.markdown("### 📷 Photo macro (optionnel)")
            st.markdown("<small style='color:#A8B4CC'>Photo macro de l'aile ou de l'abeille (si le fournisseur IA supporte la vision)</small>",
                        unsafe_allow_html=True)
            img_file = st.file_uploader("Photo macro abeille", type=["jpg","jpeg","png","webp"],
                                        key="morpho_img")

            col_btn1, col_btn2 = st.columns(2)
            btn_local  = col_btn1.button("🔬 Classifier (local)", use_container_width=True)
            btn_ia     = col_btn2.button("🤖 Analyser avec l'IA", use_container_width=True,
                                          disabled=not ia_active)

        with col2:
            st.markdown("### 📊 Résultats — Classification Ruttner 1988")
            scores     = classify_race(aile, cubital, glossa)
            race_prob  = max(scores, key=scores.get)
            confiance  = scores[race_prob]

            st.markdown(f"""
            <div style='background:#0F1117;border:1px solid #C8820A;border-left:4px solid #C8820A;
                        border-radius:8px;padding:12px 16px;margin-bottom:12px'>
                <div style='font-size:.95rem;font-weight:600;color:#F0F4FF'>
                    Race probable : <span style='color:#F5A623'>Apis mellifera {race_prob}</span>
                </div>
                <div style='font-size:.78rem;color:#A8B4CC;margin-top:3px'>
                    Algorithme local · Confiance {confiance}% ·
                    aile={aile}mm / cubital={cubital} / glossa={glossa}mm
                </div>
            </div>
            """, unsafe_allow_html=True)

            couleurs = {"intermissa":"#C8820A","sahariensis":"#8B7355",
                        "ligustica":"#2E7D32","carnica":"#1565C0","hybride":"#888"}
            fig = go.Figure()
            for race, pct in sorted(scores.items(), key=lambda x: -x[1]):
                fig.add_trace(go.Bar(y=[race], x=[pct], orientation="h",
                                     marker_color=couleurs.get(race,"#ccc"),
                                     text=f"{pct}%", textposition="auto", name=race))
            fig.update_layout(height=220, showlegend=False, template="plotly_white",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=0,b=0,l=0,r=10),
                              xaxis=dict(range=[0,100], title="Confiance (%)"))
            st.plotly_chart(fig, use_container_width=True)

            prod_scores = {
                "intermissa":   {"miel":4,"pollen":3,"propolis":5,"gr":2},
                "sahariensis":  {"miel":3,"pollen":4,"propolis":3,"gr":2},
                "ligustica":    {"miel":5,"pollen":3,"propolis":1,"gr":3},
                "carnica":      {"miel":4,"pollen":4,"propolis":2,"gr":3},
                "hybride":      {"miel":3,"pollen":3,"propolis":3,"gr":2},
            }
            ps = prod_scores.get(race_prob, {"miel":3,"pollen":3,"propolis":3,"gr":2})
            st.markdown("**Potentiel de production estimé (algorithme local) :**")
            cols_s = st.columns(4)
            for col, (label, icon, key) in zip(cols_s, [
                ("Miel","🍯","miel"), ("Pollen","🌼","pollen"),
                ("Propolis","🟤","propolis"), ("Gelée R.","👑","gr")
            ]):
                note = ps[key]
                etoiles = "⭐" * note + "☆" * (5 - note)
                col.markdown(f"<div style='text-align:center;font-size:.75rem;color:#A8B4CC'>{icon} {label}</div>"
                             f"<div style='text-align:center;font-size:.85rem'>{etoiles}</div>",
                             unsafe_allow_html=True)

        if btn_local:
            rid = opts[ruche_sel]
            conf_json = json.dumps([{"race": r, "confiance": p} for r, p in scores.items()])
            spec = " / ".join(specialisations.get(race_prob, []))
            conn.execute("""
                INSERT INTO morph_analyses
                (ruche_id,date_analyse,longueur_aile_mm,largeur_aile_mm,indice_cubital,
                 glossa_mm,tomentum,pigmentation,race_probable,confiance_json,specialisation,notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (rid, str(datetime.date.today()), aile, largeur, cubital,
                  glossa, tomentum, pigmentation, race_prob, conf_json, spec, notes))
            conn.commit()
            log_action("Morphométrie classifiée (local)", f"Ruche {ruche_sel} — {race_prob} {confiance}%")
            result_json = {
                "id_analyse": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "date": datetime.datetime.now().isoformat() + "Z",
                "ruche": ruche_sel,
                "morphometrie": {
                    "mesures": {"longueur_aile_mm": aile, "largeur_aile_mm": largeur,
                                "indice_cubital": cubital, "glossa_mm": glossa,
                                "tomentum": tomentum, "pigmentation": pigmentation},
                    "classification_raciale": [{"race": r, "confiance": p} for r, p in scores.items()],
                    "race_probable": race_prob, "specialisation": spec,
                }
            }
            st.success(f"✅ Classification locale sauvegardée : **{race_prob}** ({confiance}%)")
            st.download_button("⬇️ Télécharger JSON", json.dumps(result_json, indent=2, ensure_ascii=False),
                               f"morpho_{datetime.date.today()}.json", "application/json")

        if btn_ia:
            img_bytes = img_file.read() if img_file else None
            prov = get_active_provider()
            with st.spinner(f"🤖 {prov} analyse les données morphométriques..."):
                resultat_ia = ia_analyser_morphometrie(
                    aile, largeur, cubital, glossa, tomentum, pigmentation,
                    race_prob, confiance, img_bytes
                )
            if resultat_ia and not resultat_ia.startswith("❌"):
                afficher_resultat_ia(resultat_ia, "Analyse morphométrique approfondie — IA")
                log_action("Morphométrie IA", f"Ruche {ruche_sel} — analyse {prov} effectuée")
                rid = opts[ruche_sel]
                conf_json = json.dumps([{"race": r, "confiance": p} for r, p in scores.items()])
                spec = " / ".join(specialisations.get(race_prob, []))
                conn.execute("""
                    INSERT INTO morph_analyses
                    (ruche_id,date_analyse,longueur_aile_mm,largeur_aile_mm,indice_cubital,
                     glossa_mm,tomentum,pigmentation,race_probable,confiance_json,specialisation,notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (rid, str(datetime.date.today()), aile, largeur, cubital,
                      glossa, tomentum, pigmentation, race_prob, conf_json, spec,
                      f"[IA] {notes}"))
                conn.commit()
            elif resultat_ia:
                st.error(resultat_ia)
            else:
                st.warning("⚠️ IA non disponible. Configurez votre clé API via le sélecteur ci-dessus.")

    with tab2:
        df = pd.read_sql("""
            SELECT m.id, r.nom as ruche, m.date_analyse, m.longueur_aile_mm,
                   m.indice_cubital, m.glossa_mm, m.race_probable, m.specialisation, m.notes
            FROM morph_analyses m JOIN ruches r ON r.id=m.ruche_id
            ORDER BY m.date_analyse DESC
        """, conn)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv, "morphometrie.csv", "text/csv")
        else:
            st.info("Aucune analyse morphométrique enregistrée.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : CARTOGRAPHIE
# ════════════════════════════════════════════════════════════════════════════
def page_carto():
    st.markdown("## 🗺️ Cartographie — Zones mellifères + Analyse IA")

    ia_active = widget_cle_api()

    conn = get_db()
    tab1, tab2, tab3 = st.tabs(["🗺️ Carte & Zones", "🌿 Analyse environnement IA", "➕ Ajouter une zone"])

    with tab1:
        df_zones  = pd.read_sql("SELECT * FROM zones", conn)
        df_ruches = pd.read_sql("SELECT * FROM ruches WHERE statut='actif' AND latitude IS NOT NULL", conn)

        if FOLIUM_OK:
            center_lat = float(df_ruches["latitude"].mean()) if not df_ruches.empty else 34.88
            center_lon = float(df_ruches["longitude"].mean()) if not df_ruches.empty else 1.32
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                           tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                           attr="Google Satellite")
            couleurs_pot = {"élevé":"green","modéré":"orange","faible":"red",
                            "exceptionnel":"darkgreen","modere":"orange"}

            for _, r in df_ruches.iterrows():
                folium.Marker(
                    [r["latitude"], r["longitude"]],
                    popup=f"<b>{r['nom']}</b><br>{r['race']}<br>{r['localisation']}",
                    icon=folium.Icon(color="orange", icon="home", prefix="fa")
                ).add_to(m)

            for _, z in df_zones.iterrows():
                if z["latitude"] and z["longitude"]:
                    col_m = couleurs_pot.get(str(z["potentiel"]).lower(), "blue")
                    popup_html = f"""
                    <b>{z['nom']}</b><br>
                    Flore : {z['flore_principale']}<br>
                    NDVI : {z['ndvi']}<br>
                    Potentiel : {z['potentiel']}<br>
                    Surface : {z['superficie_ha']} ha
                    """
                    folium.CircleMarker(
                        [z["latitude"], z["longitude"]], radius=14,
                        popup=folium.Popup(popup_html, max_width=200),
                        color=col_m, fill=True, fill_color=col_m, fill_opacity=0.55
                    ).add_to(m)

            st_folium(m, width="100%", height=420)
        else:
            st.warning("Installez `folium` et `streamlit-folium` pour la carte interactive.")

        st.markdown("### 📋 Zones enregistrées")
        if not df_zones.empty:
            for _, z in df_zones.iterrows():
                with st.expander(f"📍 {z['nom']} — {z['flore_principale']} · {z['potentiel']}"):
                    col_z1, col_z2, col_z3, col_z4 = st.columns(4)
                    col_z1.metric("Surface", f"{z['superficie_ha']} ha")
                    col_z2.metric("NDVI", f"{z['ndvi']:.2f}")
                    col_z3.metric("Type", z["type_zone"])
                    col_z4.metric("Potentiel", z["potentiel"])

                    if st.button(f"🤖 Analyser '{z['nom']}' avec l'IA",
                                  key=f"ia_zone_{z['id']}", disabled=not ia_active):
                        prov = get_active_provider()
                        with st.spinner(f"🤖 {prov} analyse la zone..."):
                            result = ia_analyser_zone_carto(
                                z["nom"], z["flore_principale"],
                                z["superficie_ha"], z["ndvi"],
                                z["potentiel"], z["type_zone"],
                                z["latitude"], z["longitude"]
                            )
                        if result and "error" not in result:
                            _afficher_diagnostic_zone(result, z["nom"])
                            log_action("Analyse IA zone", f"Zone '{z['nom']}' analysée par {prov}")
                        elif result:
                            st.error(f"Erreur IA : {result.get('error')}")
                        else:
                            st.warning("⚠️ Configurez votre clé API via le sélecteur ci-dessus.")

    with tab2:
        st.markdown("### 🌿 Analyse IA d'un environnement mellifère")
        st.markdown("""
        <div style='background:#0D2A1F;border:1px solid #1A5C3A;border-radius:8px;padding:12px;
                    font-size:.83rem;color:#F0F4FF;margin-bottom:16px'>
        📸 Décrivez votre environnement (ou téléversez une photo) et l'IA évalue
        le potentiel <b>Miel / Pollen / Propolis / Gelée royale</b> sur une échelle /5 ⭐<br>
        ✅ Fonctionne avec <b>Gemma, Claude, Groq, Mistral, OpenRouter</b> et tous les fournisseurs configurés.
        </div>
        """, unsafe_allow_html=True)

        col_env1, col_env2 = st.columns([1.2, 1])
        with col_env1:
            description = st.text_area(
                "Description de l'environnement *",
                placeholder=(
                    "Ex : Zone de garrigue méditerranéenne avec chênes-lièges dominants, "
                    "romarin, lavande stoechas, thym et jujubiers en bordure. "
                    "Exposition sud, altitude 600m, oued permanent à 300m, "
                    "pas de cultures agricoles à proximité..."
                ),
                height=140,
                key="env_description"
            )
            col_s1, col_s2 = st.columns(2)
            saison = col_s1.selectbox("Saison actuelle",
                                       ["Printemps","Été","Automne","Hiver"], key="env_saison")
            col_lat, col_lon = st.columns(2)
            env_lat = col_lat.number_input("Latitude (optionnel)", -90.0, 90.0, 34.88, 0.0001,
                                            format="%.4f", key="env_lat")
            env_lon = col_lon.number_input("Longitude (optionnel)", -180.0, 180.0, 1.32, 0.0001,
                                            format="%.4f", key="env_lon")

        with col_env2:
            st.markdown("**📷 Photo du paysage / de la flore (optionnel)**")
            env_img = st.file_uploader("Photo paysage ou flore", type=["jpg","jpeg","png","webp"],
                                        key="env_img")
            if env_img:
                st.image(env_img, caption="Aperçu de l'environnement", use_container_width=True)

        prov_actif = get_active_provider()
        btn_env = st.button(f"🤖 Lancer l'analyse avec {prov_actif.split('(')[0].strip()}",
                             use_container_width=True, disabled=not ia_active)

        if not ia_active:
            st.info("🔑 Configurez votre clé API (sélecteur ci-dessus) pour activer l'analyse IA.")

        if btn_env:
            if not description.strip():
                st.warning("⚠️ Veuillez décrire l'environnement.")
            else:
                img_bytes = env_img.read() if env_img else None
                with st.spinner(f"🤖 {prov_actif} analyse l'environnement mellifère... (5-15 secondes)"):
                    resultat = ia_analyser_environnement(
                        description, env_lat, env_lon, saison, img_bytes
                    )
                if resultat and not resultat.startswith("❌"):
                    afficher_resultat_ia(resultat, "Analyse environnementale mellifère — IA")
                    log_action("Analyse IA environnement",
                               f"Zone {env_lat:.2f},{env_lon:.2f} — {saison} — {prov_actif}")

                    st.markdown("---")
                    st.markdown("**💾 Sauvegarder cette zone dans la cartographie ?**")
                    with st.form("save_env_zone"):
                        nom_z = st.text_input("Nom de la zone", "Zone analysée IA")
                        type_z = st.selectbox("Type", ["nectar","pollen","nectar+pollen","propolis","mixte"])
                        surf_z = st.number_input("Superficie estimée (ha)", 0.0, 5000.0, 10.0)
                        if st.form_submit_button("💾 Sauvegarder dans la cartographie"):
                            conn.execute("""
                                INSERT INTO zones (nom,type_zone,latitude,longitude,superficie_ha,
                                                   flore_principale,potentiel,notes)
                                VALUES (?,?,?,?,?,?,?,?)
                            """, (nom_z, type_z, env_lat, env_lon, surf_z,
                                  description[:100], "élevé", "[IA] " + description[:200]))
                            conn.commit()
                            log_action("Zone sauvegardée depuis analyse IA", nom_z)
                            st.success(f"✅ Zone '{nom_z}' sauvegardée dans la cartographie !")
                elif resultat:
                    st.error(resultat)

    with tab3:
        with st.form("add_zone"):
            col1, col2 = st.columns(2)
            nom       = col1.text_input("Nom de la zone*")
            type_zone = col2.selectbox("Type", ["nectar","pollen","nectar+pollen","propolis","mixte"])
            col3, col4 = st.columns(2)
            lat       = col3.number_input("Latitude", value=34.88, format="%.4f")
            lon       = col4.number_input("Longitude", value=1.32, format="%.4f")
            col5, col6, col7 = st.columns(3)
            superficie = col5.number_input("Superficie (ha)", 0.0, 5000.0, 10.0)
            flore      = col6.text_input("Flore principale")
            ndvi       = col7.number_input("NDVI", 0.0, 1.0, 0.65, 0.01)
            potentiel  = st.selectbox("Potentiel mellifère", ["faible","modéré","élevé","exceptionnel"])
            notes      = st.text_area("Notes")
            submitted  = st.form_submit_button("✅ Ajouter la zone")

        if submitted and nom:
            conn.execute("""
                INSERT INTO zones (nom,type_zone,latitude,longitude,superficie_ha,
                                   flore_principale,ndvi,potentiel,notes)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (nom, type_zone, lat, lon, superficie, flore, ndvi, potentiel, notes))
            conn.commit()
            log_action("Zone ajoutée", f"Zone '{nom}' — {flore} — NDVI {ndvi}")
            st.success(f"✅ Zone '{nom}' ajoutée.")
            st.rerun()

    conn.close()


def _afficher_diagnostic_zone(result, nom_zone):
    d = result.get("diagnostic", {})
    scores = result.get("scores", {})

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#F0F9F0,#1E2535);
                border:1px solid #2E7D32;border-left:4px solid #2E7D32;
                border-radius:10px;padding:16px;margin:8px 0'>
        <div style='font-family:Playfair Display,serif;font-size:.95rem;font-weight:600;
                    color:#6EE7B7;margin-bottom:10px'>🤖 Diagnostic IA — {nom_zone}</div>
        <div style='display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px'>
            <span>🌿 Potentiel : <b>{d.get('potentiel_global','—')}</b></span>
            <span>📊 Indice mellifère : <b>{d.get('indice_mellifere','—')}/10</b></span>
            <span>🐝 Capacité : <b>{d.get('capacite_ruches','—')} ruches</b></span>
            <span>📅 Pic : <b>{d.get('saison_pic','—')}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if scores:
        st.markdown("**Scores de production :**")
        cols_sc = st.columns(4)
        icons = {"miel":"🍯","pollen":"🌼","propolis":"🟤","gelee_royale":"👑"}
        labels = {"miel":"Miel","pollen":"Pollen","propolis":"Propolis","gelee_royale":"Gelée royale"}
        for col, key in zip(cols_sc, ["miel","pollen","propolis","gelee_royale"]):
            s = scores.get(key, {})
            with col:
                st.markdown(f"""
                <div style='text-align:center;background:#1E2535;border:1px solid #2E3A52;
                            border-radius:8px;padding:10px'>
                    <div style='font-size:1.2rem'>{icons[key]}</div>
                    <div style='font-size:.75rem;color:#A8B4CC;font-weight:500'>{labels[key]}</div>
                    <div style='font-size:.9rem'>{s.get('etoiles','—')}</div>
                    <div style='font-size:.7rem;color:#A8B4CC'>{s.get('detail','')[:50]}</div>
                </div>
                """, unsafe_allow_html=True)

    flore_list = result.get("flore_identifiee", [])
    if flore_list:
        st.markdown("**Flore identifiée par l'IA :**")
        df_f = pd.DataFrame(flore_list)
        st.dataframe(df_f, use_container_width=True, hide_index=True)

    recs = result.get("recommandations", [])
    if recs:
        st.markdown("**Recommandations :**")
        for r in recs:
            st.markdown(f"- {r}")

    resume = result.get("resume", "")
    if resume:
        st.info(f"📝 {resume}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE : MÉTÉO & MIELLÉE
# ════════════════════════════════════════════════════════════════════════════
def page_meteo():
    st.markdown("## ☀️ Météo & Miellée — Prévisions 7 jours")
    localisation = get_setting("localisation", "Tlemcen")
    st.markdown(f"<p style='color:#A8B4CC'>Données simulées · {localisation}</p>", unsafe_allow_html=True)

    today = datetime.date.today()
    previsions = [
        {"jour": (today + datetime.timedelta(days=i)).strftime("%a %d/%m"), "temp": t, "icon": ic, "butinage": b, "pluie": p}
        for i, (t, ic, b, p) in enumerate([
            (22, "☀️", "Élevé", 0),
            (19, "⛅", "Élevé", 5),
            (21, "🌤️", "Élevé", 10),
            (14, "🌧️", "Faible", 80),
            (17, "⛅", "Moyen", 30),
            (24, "☀️", "Élevé", 0),
            (26, "☀️", "Élevé", 0),
        ])
    ]

    cols = st.columns(7)
    couleur_butinage = {"Élevé": "#2E7D32", "Moyen": "#F57F17", "Faible": "#C62828"}
    bg_butinage = {"Élevé": "#E8F5E9", "Moyen": "#FFF8E1", "Faible": "#FFEBEE"}

    for col, p in zip(cols, previsions):
        with col:
            st.markdown(f"""
            <div style='background:#1E2535;border:1px solid #2E3A52;border-radius:8px;padding:10px 6px;text-align:center'>
                <div style='font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#A8B4CC;font-weight:500'>{p['jour']}</div>
                <div style='font-size:1.4rem;margin:4px 0'>{p['icon']}</div>
                <div style='font-size:.85rem;font-weight:500;color:#F0F4FF'>{p['temp']}°C</div>
                <div style='font-size:.65rem;margin-top:4px;padding:2px 4px;border-radius:4px;
                    background:{bg_butinage[p["butinage"]]};color:{couleur_butinage[p["butinage"]]}'>{p['butinage']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Indice de butinage prévisionnel")
        df_but = pd.DataFrame(previsions)
        indice = {"Élevé": 90, "Moyen": 55, "Faible": 15}
        df_but["indice"] = df_but["butinage"].map(indice)
        fig = px.bar(df_but, x="jour", y="indice", template="plotly_white",
                     color_discrete_sequence=["#C8820A"])
        fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=0,b=0,l=0,r=0), yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 💡 Conseils de la semaine")
        st.success("☀️ **Lundi–Jeudi** : Conditions excellentes. Prioriser inspections et pose de hausses sur R01, R02, R04.")
        st.warning("🌧️ **Vendredi** : Pluie prévue. Éviter toute intervention. Vérifier fermetures.")
        st.info("🍯 **Dimanche–Lundi** : Pic de miellée jujubier prévu. Planifier la récolte en début de semaine prochaine.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE : GÉNÉTIQUE & SÉLECTION
# ════════════════════════════════════════════════════════════════════════════
def page_genetique():
    st.markdown("## 📊 Génétique & Sélection")
    conn = get_db()

    df = pd.read_sql("""
        SELECT r.nom, r.race,
               COALESCE(AVG(i.varroa_pct), 0) as varroa_moy,
               COALESCE(AVG(i.nb_cadres), 0) as cadres_moy,
               COALESCE(SUM(rec.quantite_kg), 0) as production_totale,
               COUNT(i.id) as nb_inspections
        FROM ruches r
        LEFT JOIN inspections i ON i.ruche_id = r.id
        LEFT JOIN recoltes rec ON rec.ruche_id = r.id AND rec.type_produit='miel'
        WHERE r.statut='actif'
        GROUP BY r.id, r.nom, r.race
        ORDER BY production_totale DESC
    """, conn)

    if not df.empty:
        df["VSH_score"] = df["varroa_moy"].apply(lambda v: max(0, min(100, 100 - v * 20)))
        df["Score global"] = (
            df["production_totale"].rank(pct=True) * 40 +
            df["VSH_score"].rank(pct=True) * 35 +
            (1 - df["varroa_moy"].rank(pct=True)) * 25
        ).round(1)

        st.markdown("### 🏆 Top 3 candidates élevage")
        top3 = df.nlargest(3, "Score global")
        for i, (_, row) in enumerate(top3.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i]
            st.success(f"{medal} **{row['nom']}** ({row['race']}) — Score : {row['Score global']:.1f}/100 · VSH {row['VSH_score']:.0f}% · Production {row['production_totale']:.1f} kg")

        st.markdown("### 📋 Registre complet")
        df_display = df[["nom","race","varroa_moy","cadres_moy","production_totale","VSH_score","Score global"]].copy()
        df_display.columns = ["Ruche","Race","Varroa moy%","Cadres moy","Production kg","VSH%","Score/100"]
        df_display = df_display.round(2)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("### 🕸️ Profil de caractérisation")
        ruche_sel = st.selectbox("Choisir une ruche", df["nom"].tolist())
        row = df[df["nom"] == ruche_sel].iloc[0]
        categories = ["Production", "VSH", "Douceur", "Économie hivernale", "Propolis"]
        values = [
            min(100, row["production_totale"] * 2),
            row["VSH_score"],
            max(0, 100 - row["varroa_moy"] * 15),
            70, 60
        ]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(200,130,10,0.2)",
            line_color="#C8820A"
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
                          height=350, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : FLORE MELLIFÈRE
# ════════════════════════════════════════════════════════════════════════════
def page_flore():
    st.markdown("## 🌿 Flore mellifère — Calendrier")
    flore_data = [
        {"Espèce": "Romarin (Rosmarinus officinalis)", "Nectar": "⭐⭐⭐", "Pollen": "⭐⭐", "Propolis": "-", "Période": "Fév–Avr", "Potentiel": "Élevé"},
        {"Espèce": "Jujubier (Ziziphus lotus)", "Nectar": "⭐⭐⭐⭐", "Pollen": "⭐⭐⭐", "Propolis": "-", "Période": "Avr–Juin", "Potentiel": "Exceptionnel"},
        {"Espèce": "Chêne-liège (Quercus suber)", "Nectar": "⭐", "Pollen": "⭐⭐⭐⭐", "Propolis": "⭐⭐", "Période": "Avr–Mai", "Potentiel": "Élevé"},
        {"Espèce": "Lavande (Lavandula stoechas)", "Nectar": "⭐⭐⭐", "Pollen": "⭐⭐", "Propolis": "-", "Période": "Mai–Juil", "Potentiel": "Élevé"},
        {"Espèce": "Thym (Thymus algeriensis)", "Nectar": "⭐⭐⭐", "Pollen": "⭐⭐⭐", "Propolis": "⭐", "Période": "Mar–Juin", "Potentiel": "Élevé"},
        {"Espèce": "Eucalyptus (E. globulus)", "Nectar": "⭐⭐⭐⭐", "Pollen": "⭐⭐", "Propolis": "⭐", "Période": "Été", "Potentiel": "Élevé"},
        {"Espèce": "Caroube (Ceratonia siliqua)", "Nectar": "⭐⭐", "Pollen": "⭐⭐", "Propolis": "-", "Période": "Sep–Oct", "Potentiel": "Modéré"},
    ]
    df_flore = pd.DataFrame(flore_data)
    st.dataframe(df_flore, use_container_width=True, hide_index=True)

    st.markdown("### 📅 Calendrier de miellée")
    mois = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Aoû","Sep","Oct","Nov","Déc"]
    esp = ["Romarin","Jujubier","Chêne-liège","Lavande","Thym","Eucalyptus","Caroube"]
    activite = np.array([
        [0,3,3,2,0,0,0,0,0,0,0,0],
        [0,0,0,3,3,2,0,0,0,0,0,0],
        [0,0,0,3,3,0,0,0,0,0,0,0],
        [0,0,0,0,3,3,3,0,0,0,0,0],
        [0,0,3,3,3,2,0,0,0,0,0,0],
        [0,0,0,0,0,0,3,3,2,0,0,0],
        [0,0,0,0,0,0,0,0,3,3,0,0],
    ], dtype=float)

    fig = px.imshow(activite, labels=dict(x="Mois", y="Espèce", color="Intensité"),
                    x=mois, y=esp,
                    color_continuous_scale=[[0,"#F5EDD8"],[0.5,"#F5C842"],[1,"#C8820A"]],
                    template="plotly_white")
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10,b=10,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE : ALERTES
# ════════════════════════════════════════════════════════════════════════════
def page_alertes():
    st.markdown("## ⚠️ Alertes")
    conn = get_db()

    df_crit = pd.read_sql("""
        SELECT r.nom, i.varroa_pct, i.date_inspection, i.notes
        FROM inspections i JOIN ruches r ON r.id=i.ruche_id
        WHERE i.varroa_pct >= 3.0 AND i.date_inspection >= date('now','-7 days')
        ORDER BY i.varroa_pct DESC
    """, conn)
    df_warn = pd.read_sql("""
        SELECT r.nom, i.varroa_pct, i.date_inspection
        FROM inspections i JOIN ruches r ON r.id=i.ruche_id
        WHERE i.varroa_pct >= 2.0 AND i.varroa_pct < 3.0 AND i.date_inspection >= date('now','-7 days')
        ORDER BY i.varroa_pct DESC
    """, conn)
    df_gr = pd.read_sql("""
        SELECT r.nom, SUM(rec.quantite_kg) as total, MAX(rec.hda_pct) as hda
        FROM recoltes rec JOIN ruches r ON r.id=rec.ruche_id
        WHERE rec.type_produit='gelée royale'
        GROUP BY r.nom HAVING total > 0.3
    """, conn)

    if not df_crit.empty:
        st.markdown("### 🔴 Alertes critiques (Varroa ≥ 3%)")
        for _, row in df_crit.iterrows():
            st.error(f"🔴 **{row['nom']}** — Varroa **{row['varroa_pct']}%** le {row['date_inspection']} · Traitement immédiat requis !")

    if not df_warn.empty:
        st.markdown("### 🟡 Alertes attention (Varroa ≥ 2%)")
        for _, row in df_warn.iterrows():
            st.warning(f"🟡 **{row['nom']}** — Varroa **{row['varroa_pct']}%** le {row['date_inspection']} · Surveillance renforcée.")

    if not df_gr.empty:
        st.markdown("### 🟢 Excellentes productrices gelée royale")
        for _, row in df_gr.iterrows():
            hda_str = f" · 10-HDA {row['hda']:.1f}%" if row["hda"] else ""
            st.success(f"🟢 **{row['nom']}** — {row['total']:.2f} kg gelée royale{hda_str} → Candidate élevage sélectif")

    if df_crit.empty and df_warn.empty and df_gr.empty:
        st.info("✅ Aucune alerte active en ce moment.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : JOURNAL
# ════════════════════════════════════════════════════════════════════════════
def page_journal():
    st.markdown("## 📋 Journal d'activité")
    conn = get_db()
    df = pd.read_sql("SELECT * FROM journal ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()

    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter CSV", csv, "journal.csv", "text/csv")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Le journal est vide.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE : ADMINISTRATION
# ════════════════════════════════════════════════════════════════════════════
def page_admin():
    st.markdown("## ⚙️ Administration")
    conn = get_db()

    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Profil rucher", "🤖 Clé API IA", "🔐 Mot de passe", "💾 Base de données"])

    with tab1:
        rucher_nom = get_setting("rucher_nom", "Mon Rucher")
        localisation = get_setting("localisation", "")
        with st.form("settings_form"):
            new_nom = st.text_input("Nom du rucher", rucher_nom)
            new_loc = st.text_input("Localisation", localisation)
            submitted = st.form_submit_button("💾 Sauvegarder")
        if submitted:
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('rucher_nom',?)", (new_nom,))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('localisation',?)", (new_loc,))
            conn.commit()
            log_action("Paramètres modifiés", f"Nom: {new_nom}, Localisation: {new_loc}")
            st.success("✅ Paramètres sauvegardés.")

    with tab2:
        st.markdown("### 🤖 Gestion des fournisseurs IA — Tous gratuits")
        st.markdown("""
        <div style='background:#0F1117;border:1px solid #C8820A;border-radius:8px;padding:14px;
                    font-size:.84rem;color:#F0F4FF;margin-bottom:16px'>
        <b>ApiTrack Pro supporte 10 fournisseurs IA 100% gratuits.</b>
        Configurez une ou plusieurs clés — l'app utilisera le fournisseur actif sélectionné.
        <b>Gemma, Groq, Mistral, OpenRouter</b> fonctionnent tous sans restriction Anthropic.
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for pname, cfg in IA_PROVIDERS.items():
            key = get_api_key_for_provider(pname)
            rows.append({
                "Fournisseur": pname,
                "Modèle par défaut": cfg["default"],
                "Quota gratuit": cfg["quota"],
                "Vision": "✅" if cfg["vision"] else "❌",
                "Statut": "✅ Configuré" if key else "❌ Manquant",
            })
        df_prov = pd.DataFrame(rows)
        st.dataframe(df_prov, use_container_width=True, hide_index=True)

        st.markdown("#### 🔑 Configurer les clés API")
        prov_sel = st.selectbox("Fournisseur à configurer",
                                 list(IA_PROVIDERS.keys()), key="admin_prov_sel")
        cfg_sel = IA_PROVIDERS[prov_sel]
        key_actuelle = get_api_key_for_provider(prov_sel)

        st.markdown(f"""
        <div style='font-size:.8rem;background:#0D2A1F;border:1px solid #1A5C3A;
                    border-radius:6px;padding:10px;margin:8px 0'>
        🔗 Obtenir la clé : <a href='{cfg_sel["url"]}' target='_blank'>{cfg_sel["url"]}</a><br>
        📊 Quota : {cfg_sel['quota']}<br>
        🖼️ Vision/Photo : {'✅ Supporté' if cfg_sel['vision'] else '❌ Texte uniquement'}
        {f"<br>⚠️ {cfg_sel['note']}" if cfg_sel.get('note') else ""}
        </div>
        """, unsafe_allow_html=True)

        with st.form(f"key_form_{prov_sel}"):
            new_key = st.text_input(
                f"Clé API pour {prov_sel.split('(')[0].strip()}",
                value=key_actuelle, type="password",
                placeholder=cfg_sel.get("prefix","") + "votre-clé-ici"
            )
            sel_model_admin = st.selectbox("Modèle à utiliser", cfg_sel["models"],
                                            index=0, key="admin_model_sel")
            col_a, col_b = st.columns(2)
            save = col_a.form_submit_button("💾 Sauvegarder & Activer")
            delete = col_b.form_submit_button("🗑️ Supprimer la clé")

        if save:
            conn = get_db()
            if new_key.strip():
                conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)",
                             (cfg_sel["key"], new_key.strip()))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('ia_provider',?)", (prov_sel,))
            conn.execute("INSERT OR REPLACE INTO settings VALUES ('ia_model',?)", (sel_model_admin,))
            conn.commit()
            conn.close()
            log_action("Fournisseur IA configuré", f"{prov_sel} / {sel_model_admin}")
            st.success(f"✅ {prov_sel} configuré et activé · Modèle : {sel_model_admin}")
            st.rerun()
        if delete:
            conn = get_db()
            conn.execute("DELETE FROM settings WHERE key=?", (cfg_sel["key"],))
            conn.commit()
            conn.close()
            st.success("✅ Clé supprimée.")
            st.rerun()

        if key_actuelle:
            if st.button("🔬 Tester la connexion", key="admin_test_ia"):
                with st.spinner("Test en cours..."):
                    r = ia_call("Réponds uniquement : 'ApiTrack Pro IA OK'")
                if r and "OK" in r:
                    st.success(f"✅ {r.strip()}")
                elif r:
                    st.info(f"Réponse : {r[:300]}")
                else:
                    st.error("❌ Pas de réponse. Vérifiez la clé.")

    with tab3:
        with st.form("pwd_form"):
            old_pwd = st.text_input("Mot de passe actuel", type="password")
            new_pwd = st.text_input("Nouveau mot de passe", type="password")
            new_pwd2 = st.text_input("Confirmer le nouveau mot de passe", type="password")
            submitted = st.form_submit_button("🔐 Changer le mot de passe")
        if submitted:
            user = check_login(st.session_state.username, old_pwd)
            if not user:
                st.error("Mot de passe actuel incorrect.")
            elif new_pwd != new_pwd2:
                st.error("Les nouveaux mots de passe ne correspondent pas.")
            elif len(new_pwd) < 6:
                st.error("Le mot de passe doit faire au moins 6 caractères.")
            else:
                new_hash = hashlib.sha256(new_pwd.encode()).hexdigest()
                conn.execute("UPDATE users SET password_hash=? WHERE username=?",
                             (new_hash, st.session_state.username))
                conn.commit()
                log_action("Changement mot de passe", "Mot de passe modifié avec succès")
                st.success("✅ Mot de passe modifié.")

    with tab4:
        st.markdown("**Sauvegarde de la base**")
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                st.download_button("⬇️ Télécharger la base SQLite", f, "apitrack_backup.db", "application/octet-stream")

        st.markdown("**Statistiques**")
        tables = ["ruches", "inspections", "traitements", "recoltes", "morph_analyses", "zones", "journal"]
        stats = {}
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            stats[t] = n
        df_stats = pd.DataFrame({"Table": stats.keys(), "Enregistrements": stats.values()})
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        version = get_setting("version", "2.0.0")
        st.markdown(f"<div class='api-footer'>ApiTrack Pro v{version} · Streamlit · SQLite · © 2025</div>", unsafe_allow_html=True)

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# ROUTEUR PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
        return

    sidebar()

    page = st.session_state.get("page", "dashboard")
    router = {
        "dashboard": page_dashboard,
        "ruches": page_ruches,
        "inspections": page_inspections,
        "traitements": page_traitements,
        "productions": page_productions,
        "morpho": page_morpho,
        "carto": page_carto,
        "meteo": page_meteo,
        "genetique": page_genetique,
        "flore": page_flore,
        "alertes": page_alertes,
        "journal": page_journal,
        "admin": page_admin,
    }
    fn = router.get(page, page_dashboard)
    fn()

    st.markdown("""
    <div class='api-footer'>
        🐝 ApiTrack Pro v2.0 · Streamlit + Python + SQLite · Rucher de l'Atlas · 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████
#            APITRACK PRO v3.0 — NOUVELLES FONCTIONNALITÉS EXCLUSIVES
# ██████████████████████████████████████████████████████████████████████████
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# INIT DB v3 — Nouvelles tables
# ════════════════════════════════════════════════════════════════════════════
def init_db_v3():
    """Crée les nouvelles tables v3.0 si elles n'existent pas."""
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS comptabilite (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_op TEXT NOT NULL,
        type_op TEXT NOT NULL CHECK(type_op IN ('recette','depense')),
        categorie TEXT NOT NULL,
        description TEXT,
        montant REAL NOT NULL,
        ruche_id INTEGER REFERENCES ruches(id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS taches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        titre TEXT NOT NULL,
        description TEXT,
        ruche_id INTEGER REFERENCES ruches(id),
        date_echeance TEXT NOT NULL,
        priorite TEXT DEFAULT 'normale' CHECK(priorite IN ('urgente','haute','normale','faible')),
        statut TEXT DEFAULT 'en_attente' CHECK(statut IN ('en_attente','en_cours','terminee','annulee')),
        categorie TEXT DEFAULT 'inspection',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS analyses_miel (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id),
        date_analyse TEXT NOT NULL,
        humidite_pct REAL,
        conductivite_ms REAL,
        couleur TEXT,
        cristallisation TEXT,
        aromes TEXT,
        origine_florale TEXT,
        score_qualite INTEGER,
        label_propose TEXT,
        ia_analyse TEXT,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS alertes_config (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type_alerte TEXT NOT NULL,
        seuil REAL,
        actif INTEGER DEFAULT 1,
        description TEXT
    );
    """)

    # Insérer alertes par défaut si table vide
    n = c.execute("SELECT COUNT(*) FROM alertes_config").fetchone()[0]
    if n == 0:
        alertes_defaut = [
            ("varroa_critique", 3.0, 1, "Varroa ≥ 3% — Traitement immédiat"),
            ("varroa_attention", 2.0, 1, "Varroa ≥ 2% — Surveillance renforcée"),
            ("poids_faible", 15.0, 1, "Poids ruche < 15 kg — Vérifier provisions"),
            ("cadres_faible", 6, 1, "Moins de 6 cadres couvain — Population en déclin"),
            ("inspection_retard", 21, 1, "Pas d'inspection depuis 21 jours"),
            ("traitement_fin", 3, 1, "Traitement se termine dans 3 jours"),
        ]
        for a in alertes_defaut:
            c.execute("INSERT INTO alertes_config (type_alerte, seuil, actif, description) VALUES (?,?,?,?)", a)

    # Insérer données démo comptabilité
    n2 = c.execute("SELECT COUNT(*) FROM comptabilite").fetchone()[0]
    if n2 == 0:
        today = datetime.date.today()
        ops_demo = [
            (str(today - datetime.timedelta(days=30)), "recette", "Vente miel", "Vente 25 kg miel toutes fleurs", 3750.0, 1),
            (str(today - datetime.timedelta(days=20)), "recette", "Vente pollen", "Pollen séché premium 2 kg", 900.0, None),
            (str(today - datetime.timedelta(days=15)), "depense", "Traitement", "Acide oxalique 250g", 450.0, None),
            (str(today - datetime.timedelta(days=10)), "depense", "Matériel", "Hausse + cadres bois", 1800.0, None),
            (str(today - datetime.timedelta(days=5)), "recette", "Vente gelée royale", "Gelée royale 150g", 2100.0, 4),
            (str(today), "depense", "Alimentation", "Sirop 10L x3 ruches", 360.0, None),
        ]
        for op in ops_demo:
            c.execute("INSERT INTO comptabilite (date_op,type_op,categorie,description,montant,ruche_id) VALUES (?,?,?,?,?,?)", op)

    # Insérer tâches démo
    n3 = c.execute("SELECT COUNT(*) FROM taches").fetchone()[0]
    if n3 == 0:
        today = datetime.date.today()
        taches_demo = [
            ("Inspection Varroa R07", "Inspection urgente — Varroa 3.8%", 6, str(today + datetime.timedelta(days=1)), "urgente", "en_attente", "traitement"),
            ("Traitement acide oxalique", "Préparer traitement 3 ruches", None, str(today + datetime.timedelta(days=3)), "haute", "en_attente", "traitement"),
            ("Récolte miel printemps", "Extraction hausse Zitoun A + Atlas C", 1, str(today + datetime.timedelta(days=7)), "normale", "en_attente", "recolte"),
            ("Contrôle reine R03", "Reine introuvable dernière inspection", 3, str(today + datetime.timedelta(days=2)), "haute", "en_attente", "inspection"),
            ("Nourrissement hivernal", "Préparer sirop pour l'automne", None, str(today + datetime.timedelta(days=30)), "faible", "en_attente", "alimentation"),
        ]
        for t in taches_demo:
            c.execute("INSERT INTO taches (titre,description,ruche_id,date_echeance,priorite,statut,categorie) VALUES (?,?,?,?,?,?,?)", t)

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : COMPTABILITÉ APICOLE
# ════════════════════════════════════════════════════════════════════════════
def page_comptabilite():
    st.markdown("## 💰 Comptabilité Apicole")
    st.markdown("<p style='color:#A8B4CC;margin-top:-10px'>Suivi financier complet — Recettes · Dépenses · ROI par ruche</p>", unsafe_allow_html=True)

    conn = get_db()

    # KPIs financiers
    total_rec = conn.execute("SELECT COALESCE(SUM(montant),0) FROM comptabilite WHERE type_op='recette'").fetchone()[0]
    total_dep = conn.execute("SELECT COALESCE(SUM(montant),0) FROM comptabilite WHERE type_op='depense'").fetchone()[0]
    benefice   = total_rec - total_dep
    nb_ruches  = conn.execute("SELECT COUNT(*) FROM ruches WHERE statut='actif'").fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Recettes totales", f"{total_rec:,.0f} DA", "+18% vs N-1")
    col2.metric("📤 Dépenses totales", f"{total_dep:,.0f} DA")
    col3.metric("📊 Bénéfice net", f"{benefice:,.0f} DA",
                delta=f"{benefice:+.0f}", delta_color="normal" if benefice >= 0 else "inverse")
    col4.metric("🐝 ROI/ruche", f"{(benefice/nb_ruches if nb_ruches else 0):,.0f} DA")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tableau de bord", "📋 Registre", "➕ Nouvelle opération", "📊 Analyse IA"])

    with tab1:
        df_mensuel = pd.read_sql("""
            SELECT strftime('%Y-%m', date_op) as mois, type_op, SUM(montant) as total
            FROM comptabilite GROUP BY mois, type_op ORDER BY mois
        """, conn)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("### 📊 Flux financiers mensuels")
            if not df_mensuel.empty:
                fig = px.bar(df_mensuel, x="mois", y="total", color="type_op",
                             color_discrete_map={"recette":"#34D399","depense":"#F87171"},
                             barmode="group", template="plotly_white")
                fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=0,r=0),
                                  legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée financière.")

        with col_g2:
            st.markdown("### 🍩 Répartition des dépenses")
            df_cat = pd.read_sql("""
                SELECT categorie, SUM(montant) as total
                FROM comptabilite WHERE type_op='depense'
                GROUP BY categorie ORDER BY total DESC
            """, conn)
            if not df_cat.empty:
                fig2 = px.pie(df_cat, values="total", names="categorie",
                              color_discrete_sequence=["#C8820A","#F5A623","#FFD07A","#8B7355","#3A4A66"],
                              template="plotly_white")
                fig2.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=0,r=0))
                st.plotly_chart(fig2, use_container_width=True)

        # ROI par ruche
        st.markdown("### 🏆 ROI par ruche")
        df_roi = pd.read_sql("""
            SELECT r.nom, r.race,
                   COALESCE((SELECT SUM(c.montant) FROM comptabilite c WHERE c.ruche_id=r.id AND c.type_op='recette'),0) as recettes,
                   COALESCE((SELECT SUM(c.montant) FROM comptabilite c WHERE c.ruche_id=r.id AND c.type_op='depense'),0) as depenses
            FROM ruches r WHERE r.statut='actif'
        """, conn)
        if not df_roi.empty:
            df_roi["benefice"] = df_roi["recettes"] - df_roi["depenses"]
            df_roi["ROI%"] = df_roi.apply(
                lambda r: round((r["recettes"] / r["depenses"] - 1) * 100, 1) if r["depenses"] > 0 else 0, axis=1)
            df_roi.columns = ["Ruche","Race","Recettes DA","Dépenses DA","Bénéfice DA","ROI%"]
            st.dataframe(df_roi.style.background_gradient(subset=["Bénéfice DA","ROI%"],
                         cmap="RdYlGn"), use_container_width=True, hide_index=True)

    with tab2:
        df_all = pd.read_sql("""
            SELECT c.id, c.date_op, c.type_op, c.categorie, c.description,
                   c.montant, COALESCE(r.nom,'—') as ruche
            FROM comptabilite c
            LEFT JOIN ruches r ON r.id=c.ruche_id
            ORDER BY c.date_op DESC
        """, conn)
        if not df_all.empty:
            csv = df_all.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv, "comptabilite.csv", "text/csv")
            st.dataframe(df_all, use_container_width=True, hide_index=True)

            # Supprimer une opération
            st.markdown("---")
            op_ids = df_all["id"].tolist()
            if op_ids:
                sel_id = st.selectbox("Supprimer une opération (ID)", op_ids)
                if st.button("🗑️ Supprimer", type="secondary"):
                    conn.execute("DELETE FROM comptabilite WHERE id=?", (sel_id,))
                    conn.commit()
                    log_action("Comptabilité", f"Opération {sel_id} supprimée")
                    st.success("✅ Opération supprimée.")
                    st.rerun()

    with tab3:
        with st.form("add_operation"):
            col1, col2 = st.columns(2)
            type_op = col1.selectbox("Type", ["recette","depense"])
            date_op = col2.date_input("Date", datetime.date.today())

            cats_rec = ["Vente miel","Vente pollen","Vente propolis","Vente gelée royale","Vente cire","Autre recette"]
            cats_dep = ["Matériel","Traitement","Alimentation","Transport","Formation","Autre dépense"]
            categorie = col1.selectbox("Catégorie", cats_rec if type_op=="recette" else cats_dep)
            montant = col2.number_input("Montant (DA)", min_value=0.0, value=500.0, step=50.0)

            description = st.text_input("Description")

            ruches_list = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
            opts_ruches = {"— Aucune ruche —": None}
            opts_ruches.update({r[1]: r[0] for r in ruches_list})
            ruche_sel = st.selectbox("Ruche associée (optionnel)", list(opts_ruches.keys()))

            submitted = st.form_submit_button("✅ Enregistrer l'opération")

        if submitted and montant > 0:
            rid = opts_ruches[ruche_sel]
            conn.execute("""
                INSERT INTO comptabilite (date_op,type_op,categorie,description,montant,ruche_id)
                VALUES (?,?,?,?,?,?)
            """, (str(date_op), type_op, categorie, description, montant, rid))
            conn.commit()
            log_action("Comptabilité", f"{type_op.capitalize()} {montant} DA — {categorie}")
            st.success(f"✅ {type_op.capitalize()} de {montant:,.0f} DA enregistrée !")
            st.rerun()

    with tab4:
        st.markdown("### 🤖 Analyse financière IA")
        ia_active = widget_ia_selector()

        if ia_active:
            df_comp = pd.read_sql("""
                SELECT type_op, categorie, SUM(montant) as total
                FROM comptabilite GROUP BY type_op, categorie ORDER BY type_op, total DESC
            """, conn)

            if not df_comp.empty and st.button("🤖 Analyser ma rentabilité", use_container_width=True):
                resume = df_comp.to_string()
                prompt = f"""Tu es consultant financier spécialisé en apiculture. Voici les données financières d'un rucher :

{resume}

Recettes totales : {total_rec:.0f} DA | Dépenses totales : {total_dep:.0f} DA | Bénéfice : {benefice:.0f} DA
Nombre de ruches actives : {nb_ruches}

Effectue une analyse financière apicole complète en français :

## 1. Diagnostic financier
- Rentabilité globale (%) et comparaison secteur
- Coût de revient par kg de miel estimé
- Revenu moyen par ruche

## 2. Points forts et points faibles
- Identifier les sources de revenus à développer
- Identifier les dépenses à optimiser

## 3. Recommandations (5 actions concrètes)
Pour améliorer la rentabilité de 20-30% dans les 12 prochains mois

## 4. Plan de diversification
- Produits à fort potentiel pour ce rucher
- Prix de vente recommandés (marché algérien 2025)

Sois précis avec des chiffres concrets."""

                with st.spinner("🤖 Analyse financière en cours..."):
                    result = ia_call(prompt)
                if result and not result.startswith("❌"):
                    afficher_resultat_ia(result, "Analyse financière apicole — IA")
                    log_action("Analyse IA comptabilité", "Analyse rentabilité effectuée")
                elif result:
                    st.error(result)

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : AGENDA & TÂCHES
# ════════════════════════════════════════════════════════════════════════════
def page_agenda():
    st.markdown("## 📆 Agenda & Tâches")
    st.markdown("<p style='color:#A8B4CC;margin-top:-10px'>Planification · Rappels · Suivi des interventions</p>", unsafe_allow_html=True)

    conn = get_db()
    today = datetime.date.today()

    # KPIs tâches
    urgentes = conn.execute("SELECT COUNT(*) FROM taches WHERE priorite='urgente' AND statut='en_attente'").fetchone()[0]
    ce_semaine = conn.execute(
        "SELECT COUNT(*) FROM taches WHERE date_echeance <= ? AND statut='en_attente'",
        (str(today + datetime.timedelta(days=7)),)
    ).fetchone()[0]
    terminees = conn.execute("SELECT COUNT(*) FROM taches WHERE statut='terminee'").fetchone()[0]
    en_retard = conn.execute(
        "SELECT COUNT(*) FROM taches WHERE date_echeance < ? AND statut NOT IN ('terminee','annulee')",
        (str(today),)
    ).fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Urgentes", urgentes)
    col2.metric("📅 Cette semaine", ce_semaine)
    col3.metric("✅ Terminées", terminees)
    col4.metric("⏰ En retard", en_retard, delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Tâches actives", "➕ Nouvelle tâche", "✅ Historique"])

    with tab1:
        # Filtre priorité
        col_f1, col_f2 = st.columns(2)
        filtre_prio = col_f1.selectbox("Filtrer par priorité", ["Toutes","urgente","haute","normale","faible"])
        filtre_cat  = col_f2.selectbox("Filtrer par catégorie", ["Toutes","inspection","traitement","recolte","alimentation","autre"])

        query = """
            SELECT t.id, t.titre, t.priorite, t.date_echeance, t.categorie, t.statut,
                   COALESCE(r.nom,'—') as ruche, t.description
            FROM taches t LEFT JOIN ruches r ON r.id=t.ruche_id
            WHERE t.statut NOT IN ('terminee','annulee')
        """
        params = []
        if filtre_prio != "Toutes":
            query += " AND t.priorite=?"
            params.append(filtre_prio)
        if filtre_cat != "Toutes":
            query += " AND t.categorie=?"
            params.append(filtre_cat)
        query += " ORDER BY CASE t.priorite WHEN 'urgente' THEN 1 WHEN 'haute' THEN 2 WHEN 'normale' THEN 3 ELSE 4 END, t.date_echeance"

        df_taches = pd.read_sql(query, conn, params=params)

        if not df_taches.empty:
            for _, t in df_taches.iterrows():
                echeance = datetime.date.fromisoformat(t["date_echeance"])
                delta_j   = (echeance - today).days
                retard    = delta_j < 0

                prio_colors = {"urgente":"#F87171","haute":"#FBD147","normale":"#60A5FA","faible":"#A8B4CC"}
                prio_icons  = {"urgente":"🔴","haute":"🟡","normale":"🔵","faible":"⚪"}
                cat_icons   = {"inspection":"🔍","traitement":"💊","recolte":"🍯","alimentation":"🌾","autre":"📌"}

                echeance_str = f"{'⏰ EN RETARD ' if retard else ''}{t['date_echeance']}"
                color_border = "#F87171" if retard else prio_colors.get(t["priorite"], "#3A4A66")

                with st.expander(f"{prio_icons.get(t['priorite'],'•')} {cat_icons.get(t['categorie'],'📌')} **{t['titre']}** — {echeance_str} · Ruche: {t['ruche']}"):
                    st.markdown(f"<p style='color:#A8B4CC;font-size:.85rem'>{t['description'] or 'Aucune description.'}</p>", unsafe_allow_html=True)

                    col_a, col_b, col_c, col_d = st.columns(4)
                    if col_a.button("✅ Terminer", key=f"done_{t['id']}"):
                        conn.execute("UPDATE taches SET statut='terminee' WHERE id=?", (t["id"],))
                        conn.commit()
                        log_action("Tâche terminée", t["titre"])
                        st.rerun()
                    if col_b.button("▶️ En cours", key=f"wip_{t['id']}"):
                        conn.execute("UPDATE taches SET statut='en_cours' WHERE id=?", (t["id"],))
                        conn.commit()
                        st.rerun()
                    if col_c.button("❌ Annuler", key=f"cancel_{t['id']}"):
                        conn.execute("UPDATE taches SET statut='annulee' WHERE id=?", (t["id"],))
                        conn.commit()
                        st.rerun()
                    col_d.markdown(f"<div style='font-size:.75rem;color:#6B7A99;padding-top:8px'>ID #{t['id']}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ Aucune tâche en attente — Rucher bien géré !")

    with tab2:
        with st.form("add_tache"):
            titre = st.text_input("Titre de la tâche *")
            description = st.text_area("Description (optionnel)", height=80)
            col1, col2, col3 = st.columns(3)
            priorite   = col1.selectbox("Priorité", ["urgente","haute","normale","faible"], index=2)
            categorie  = col2.selectbox("Catégorie", ["inspection","traitement","recolte","alimentation","autre"])
            date_ech   = col3.date_input("Échéance", today + datetime.timedelta(days=7))

            ruches_list = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
            opts_r = {"— Aucune ruche —": None}
            opts_r.update({r[1]: r[0] for r in ruches_list})
            ruche_sel = st.selectbox("Ruche associée (optionnel)", list(opts_r.keys()))

            submitted = st.form_submit_button("✅ Créer la tâche")

        if submitted and titre:
            rid = opts_r[ruche_sel]
            conn.execute("""
                INSERT INTO taches (titre,description,ruche_id,date_echeance,priorite,categorie)
                VALUES (?,?,?,?,?,?)
            """, (titre, description, rid, str(date_ech), priorite, categorie))
            conn.commit()
            log_action("Tâche créée", f"{titre} — {priorite} — {date_ech}")
            st.success(f"✅ Tâche '{titre}' créée pour le {date_ech} !")
            st.rerun()

        # Génération IA de tâches automatiques
        st.markdown("---")
        st.markdown("### 🤖 Générer des tâches automatiquement avec l'IA")
        ia_ok = get_api_key_for_provider(get_active_provider())
        if ia_ok:
            if st.button("🤖 Analyser le rucher et proposer des tâches", use_container_width=True):
                conn2 = get_db()
                df_insp = pd.read_sql("""
                    SELECT r.nom, i.varroa_pct, i.date_inspection, i.comportement, i.nb_cadres, i.reine_vue
                    FROM inspections i JOIN ruches r ON r.id=i.ruche_id
                    WHERE i.date_inspection >= date('now','-14 days')
                    ORDER BY i.date_inspection DESC
                """, conn2)
                conn2.close()

                prompt = f"""Tu es expert apicole. Voici les inspections récentes :
{df_insp.to_string() if not df_insp.empty else 'Aucune inspection récente'}
Date aujourd'hui : {today}

Génère une liste de 5 tâches prioritaires en JSON UNIQUEMENT (pas de texte avant/après) :
[
  {{"titre":"...","description":"...","priorite":"urgente|haute|normale|faible","categorie":"inspection|traitement|recolte|alimentation|autre","jours_echeance":3}},
  ...
]"""
                with st.spinner("🤖 Génération des tâches..."):
                    result = ia_call(prompt, json_mode=True)

                if result and not result.startswith("❌"):
                    import re
                    try:
                        text = result.strip()
                        m = re.search(r'\[.*\]', text, re.DOTALL)
                        if m:
                            taches_ia = json.loads(m.group())
                            st.markdown("#### 📋 Tâches proposées par l'IA :")
                            for t_ia in taches_ia:
                                echeance_ia = today + datetime.timedelta(days=t_ia.get("jours_echeance", 7))
                                st.markdown(f"- **{t_ia.get('titre','')}** ({t_ia.get('priorite','normale')}) — {echeance_ia}")
                            if st.button("💾 Importer toutes ces tâches", key="import_ia_tasks"):
                                for t_ia in taches_ia:
                                    echeance_ia = today + datetime.timedelta(days=t_ia.get("jours_echeance", 7))
                                    conn.execute("""
                                        INSERT INTO taches (titre,description,date_echeance,priorite,categorie)
                                        VALUES (?,?,?,?,?)
                                    """, (t_ia.get("titre","Tâche IA"),
                                          t_ia.get("description",""),
                                          str(echeance_ia),
                                          t_ia.get("priorite","normale"),
                                          t_ia.get("categorie","autre")))
                                conn.commit()
                                log_action("Import tâches IA", f"{len(taches_ia)} tâches importées")
                                st.success(f"✅ {len(taches_ia)} tâches importées !")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Erreur parsing IA : {e}")
                elif result:
                    st.error(result)
        else:
            st.info("🔑 Configurez une clé IA dans Administration pour activer la génération automatique.")

    with tab3:
        df_hist = pd.read_sql("""
            SELECT t.id, t.titre, t.priorite, t.date_echeance, t.statut, t.categorie,
                   COALESCE(r.nom,'—') as ruche
            FROM taches t LEFT JOIN ruches r ON r.id=t.ruche_id
            WHERE t.statut IN ('terminee','annulee')
            ORDER BY t.date_echeance DESC LIMIT 50
        """, conn)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune tâche terminée.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : ANALYSEUR DE MIEL IA
# ════════════════════════════════════════════════════════════════════════════
def page_analyseur_miel():
    st.markdown("## 🍯 Analyseur de Miel IA")
    st.markdown("""
    <p style='color:#A8B4CC;margin-top:-10px'>
    Analyse qualité · Détection falsification · Label & AOC · Score nutritionnel
    </p>
    """, unsafe_allow_html=True)

    conn = get_db()
    ia_active = widget_ia_selector()

    tab1, tab2 = st.tabs(["🔬 Nouvelle analyse", "📋 Historique analyses"])

    with tab1:
        st.markdown("""
        <div style='background:#0D2A1F;border:1px solid #1A5C3A;border-radius:8px;padding:14px;
                    font-size:.83rem;color:#F0F4FF;margin-bottom:16px'>
        🔬 <b>Analyseur de miel unique au monde</b> — Entrez les paramètres de votre miel et l'IA évalue :
        qualité, origine florale, risque de falsification, label proposé (AO, Bio, Premium), et score nutritionnel.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            ruches_list = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
            opts_r = {r[1]: r[0] for r in ruches_list}
            ruche_sel = st.selectbox("Ruche source *", list(opts_r.keys())) if opts_r else None

            humidite = st.slider("💧 Humidité (%)", 14.0, 22.0, 17.2, 0.1)
            conductivite = st.number_input("⚡ Conductivité électrique (mS/cm)", 0.0, 3.0, 0.4, 0.01)
            ph = st.number_input("🧪 pH", 3.0, 6.0, 3.8, 0.1)
            hda = st.number_input("🌿 10-HDA % (gelée royale uniquement, 0=absent)", 0.0, 6.0, 0.0, 0.1)

        with col2:
            couleur = st.selectbox("🎨 Couleur", ["Blanc eau (Extra White)","Blanc (White)","Extra Ambré clair (Extra Light Amber)",
                                                    "Ambré clair (Light Amber)","Ambré (Amber)","Ambré foncé (Dark Amber)","Foncé (Dark)"])
            cristallisation = st.selectbox("🔮 État de cristallisation", ["Liquide","Partiellement cristallisé","Totalement cristallisé","Crémeux"])
            aromes = st.text_input("👃 Arômes perçus", placeholder="Floral, fruité, épicé, boisé, caramel...")
            origine_florale = st.text_input("🌸 Origine florale supposée",
                                             placeholder="Jujubier, romarin, lavande, toutes fleurs...")
            date_recolte_miel = st.date_input("📅 Date de récolte", datetime.date.today())
            notes_miel = st.text_area("📝 Notes complémentaires", height=70,
                                       placeholder="Conditions de stockage, zone de production, observations...")

            photo_miel = st.file_uploader("📷 Photo du miel (optionnel)", type=["jpg","jpeg","png"])
            if photo_miel:
                st.image(photo_miel, caption="Aperçu", width=200)

        col_b1, col_b2 = st.columns(2)
        btn_local  = col_b1.button("🔬 Analyse rapide (locale)", use_container_width=True)
        btn_ia_miel = col_b2.button("🤖 Analyse IA approfondie", use_container_width=True, disabled=not ia_active)

        # ── Analyse locale ──────────────────────────────────────────────────
        if btn_local:
            # Score qualité heuristique
            score = 100
            issues = []
            if humidite > 19.0:
                score -= 20; issues.append(f"⚠️ Humidité élevée ({humidite}%) — risque fermentation")
            elif humidite > 17.5:
                score -= 5; issues.append(f"📌 Humidité légèrement haute ({humidite}%)")
            if humidite < 15.5:
                score -= 5; issues.append("📌 Humidité très basse — vérifier surmaturation")

            if ph < 3.4 or ph > 4.5:
                score -= 10; issues.append(f"⚠️ pH inhabituel ({ph}) — possible falsification acide")

            if conductivite > 2.0:
                score -= 10; issues.append("⚠️ Conductivité élevée — miel de miellat ou mélange probable")

            label = "Premium ⭐⭐⭐" if score >= 90 else ("Qualité A ⭐⭐" if score >= 75 else ("Qualité B ⭐" if score >= 60 else "Non conforme ⚠️"))

            st.markdown(f"""
            <div style='background:#0F1117;border:1px solid #C8820A;border-left:4px solid #C8820A;
                        border-radius:8px;padding:16px;margin:12px 0'>
                <div style='font-size:1.1rem;font-weight:700;color:#F5A623;margin-bottom:8px'>
                    Score qualité : {score}/100 — {label}
                </div>
                <div style='font-size:.85rem;color:#F0F4FF'>
                    Humidité : {"✅ Conforme" if 14.5 <= humidite <= 19.0 else "❌ Hors norme"} ({humidite}%) &nbsp;|&nbsp;
                    pH : {"✅" if 3.4 <= ph <= 4.5 else "⚠️"} ({ph}) &nbsp;|&nbsp;
                    Conductivité : {"✅" if conductivite <= 0.8 else "⚠️"} ({conductivite} mS/cm)
                </div>
                {"".join(f'<div style=\"font-size:.8rem;color:#FBD147;margin-top:6px\">{iss}</div>' for iss in issues) if issues else '<div style=\"color:#34D399;margin-top:6px;font-size:.85rem\">✅ Tous les paramètres sont conformes aux normes européennes.</div>'}
            </div>
            """, unsafe_allow_html=True)

            # Sauvegarder
            if ruche_sel and opts_r:
                rid = opts_r[ruche_sel]
                conn.execute("""
                    INSERT INTO analyses_miel (ruche_id,date_analyse,humidite_pct,conductivite_ms,couleur,
                                               cristallisation,aromes,origine_florale,score_qualite,label_propose,notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (rid, str(date_recolte_miel), humidite, conductivite, couleur,
                      cristallisation, aromes, origine_florale, score, label, notes_miel))
                conn.commit()
                log_action("Analyse miel locale", f"Ruche {ruche_sel} — Score {score}/100")
                st.success(f"✅ Analyse sauvegardée — Score {score}/100 — {label}")

        # ── Analyse IA ──────────────────────────────────────────────────────
        if btn_ia_miel:
            img_bytes = photo_miel.read() if photo_miel else None
            prompt = f"""Tu es expert en mélisopalynologie, chimie du miel et certification apicole. Voici les paramètres d'un miel algérien à analyser :

**Ruche source :** {ruche_sel or 'Non spécifiée'}
**Date récolte :** {date_recolte_miel}
**Paramètres physico-chimiques :**
- Humidité : {humidite}%
- Conductivité électrique : {conductivite} mS/cm
- pH : {ph}
- 10-HDA (gelée royale) : {hda}%
- Couleur : {couleur}
- État : {cristallisation}
- Arômes : {aromes or 'Non renseignés'}
- Origine florale déclarée : {origine_florale or 'Non spécifiée'}
- Notes : {notes_miel or 'Aucune'}

Effectue une analyse complète en français :

## 🔬 1. Conformité aux normes (Codex Alimentarius + Directive 2001/110/CE)
- Humidité : norme < 20% — Évaluation
- pH : norme 3.2–4.5 — Évaluation  
- Conductivité : norme < 0.8 mS/cm (fleurs) ou > 0.8 (miellat) — Évaluation
- Conclusion de conformité

## 🌸 2. Identification de l'origine florale
- Espèce(s) probable(s) basées sur les paramètres
- Profil pollinique attendu
- Saison de récolte probable

## 🚨 3. Détection de falsification
- Risque d'adultération (sucres, eau, HMF élevé)
- Score de risque : Faible / Modéré / Élevé
- Indicateurs suspects identifiés

## 🏅 4. Label et certification proposés
- Label qualité : Premium / Qualité A / Qualité B / Non conforme
- Score global /100 avec pondération (humidité 30%, conductivité 25%, pH 20%, couleur/cristallisation 25%)
- Certifications possibles : Bio, AO, IGP, Label Rouge (critères à remplir)
- Prix de vente conseillé (marché algérien 2025)

## 🍽️ 5. Profil nutritionnel estimé
- Glucides, vitamines, minéraux principaux probables
- Indice glycémique estimé
- Propriétés médicinales connues pour cette origine

## 📦 6. Recommandations stockage et conditionnement
- Conditions idéales de conservation
- Durée de vie estimée
- Conditionnement recommandé (pot verre, étiquetage)

Donne des chiffres précis et des références normatives (Codex, EU)."""

            with st.spinner("🤖 Analyse approfondie du miel en cours..."):
                result = ia_call(prompt, img_bytes)

            if result and not result.startswith("❌"):
                afficher_resultat_ia(result, "Analyse qualité miel — IA Expert")

                # Extraire le score depuis le résultat IA (heuristique simple)
                import re
                score_match = re.search(r'Score.*?(\d{2,3})/100', result)
                score_ia = int(score_match.group(1)) if score_match else 80
                label_match = re.search(r'Label.*?:(.*?)(?:\n|\.)', result)
                label_ia = label_match.group(1).strip() if label_match else "Qualité A"

                if ruche_sel and opts_r:
                    rid = opts_r[ruche_sel]
                    conn.execute("""
                        INSERT INTO analyses_miel (ruche_id,date_analyse,humidite_pct,conductivite_ms,couleur,
                                                   cristallisation,aromes,origine_florale,score_qualite,label_propose,ia_analyse,notes)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (rid, str(date_recolte_miel), humidite, conductivite, couleur,
                          cristallisation, aromes, origine_florale, score_ia, label_ia,
                          result[:2000], notes_miel))
                    conn.commit()
                    log_action("Analyse miel IA", f"Ruche {ruche_sel} — {get_active_provider()}")
                    st.success("✅ Analyse IA sauvegardée dans l'historique !")
            elif result:
                st.error(result)

    with tab2:
        df_am = pd.read_sql("""
            SELECT am.id, COALESCE(r.nom,'—') as ruche, am.date_analyse, am.humidite_pct,
                   am.conductivite_ms, am.origine_florale, am.score_qualite, am.label_propose, am.notes
            FROM analyses_miel am LEFT JOIN ruches r ON r.id=am.ruche_id
            ORDER BY am.date_analyse DESC
        """, conn)
        if not df_am.empty:
            st.dataframe(df_am, use_container_width=True, hide_index=True)
            csv = df_am.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", csv, "analyses_miel.csv", "text/csv")
        else:
            st.info("Aucune analyse de miel enregistrée.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : MÉTÉO & PRÉDICTIONS MELLIFÈRES AMÉLIORÉE
# ════════════════════════════════════════════════════════════════════════════
def page_meteo():
    """Version améliorée avec calendrier prédictif mellifère."""
    st.markdown("## ☀️ Météo & Calendrier Mellifère Prédictif")
    st.markdown("<p style='color:#A8B4CC;margin-top:-10px'>Prévisions · Activité butineuses · Planning automatique par mois</p>", unsafe_allow_html=True)

    ia_active = widget_ia_selector()

    localisation = get_setting("localisation", "Tlemcen, Algérie")
    st.markdown(f"📍 Rucher : **{get_setting('rucher_nom','Mon Rucher')}** — {localisation}")

    tab1, tab2, tab3 = st.tabs(["📅 Calendrier mellifère", "☀️ Météo manuelle", "🤖 Prédiction IA saisonnière"])

    with tab1:
        st.markdown("### 📅 Calendrier mellifère interactif — Tlemcen")
        mois_labels = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Aoû","Sep","Oct","Nov","Déc"]
        mois_selected = st.select_slider("Sélectionner un mois", options=mois_labels,
                                          value=mois_labels[datetime.date.today().month - 1])
        idx_mois = mois_labels.index(mois_selected)

        # Base de données calendrier mellifère Tlemcen
        calendrier = {
            0:  {"temp_min":5,"temp_max":14,"pluie":60,"floraison":["Arbousier (fin)","Eucalyptus"],"activite":20,"conseil":"Maintenir couvercles isolants. Vérifier provisions stocks. Réduire entrées."},
            1:  {"temp_min":6,"temp_max":15,"pluie":55,"floraison":["Romarin","Amandier","Eucalyptus"],"activite":35,"conseil":"Premières sorties des butineuses. Contrôle entrée de ruche. Préparer matériel."},
            2:  {"temp_min":8,"temp_max":18,"pluie":50,"floraison":["Romarin","Amandier","Pêcher","Aubépine"],"activite":65,"conseil":"🟡 Mois clé ! Visiter toutes les ruches. Stimuler avec sirop si nécessaire."},
            3:  {"temp_min":11,"temp_max":21,"pluie":40,"floraison":["Jujubier","Chêne-liège","Thym","Agrumes"],"activite":90,"conseil":"🔥 Pic de miellée printemps ! Ajouter hausses. Surveiller essaimage."},
            4:  {"temp_min":15,"temp_max":26,"pluie":25,"floraison":["Jujubier","Lavande","Thym","Ronce"],"activite":95,"conseil":"🔥 Miellée maximale ! Récolter miel printemps fin mai/début juin."},
            5:  {"temp_min":19,"temp_max":31,"pluie":12,"floraison":["Lavande","Eucalyptus","Ronce","Caroube"],"activite":75,"conseil":"Surveiller chaleur. Ombrager les ruches. Récolte fin juin."},
            6:  {"temp_min":23,"temp_max":36,"pluie":5,"floraison":["Eucalyptus","Garrigue","Lavande coton"],"activite":45,"conseil":"⚠️ Canicule — Abreuvoir obligatoire. Réduire ouverture d'entrée (prédateurs)."},
            7:  {"temp_min":23,"temp_max":36,"pluie":8,"floraison":["Eucalyptus","Garrigue sec"],"activite":30,"conseil":"⚠️ Stress hydrique. Nourrir si nécessaire. Traitement varroa acide oxalique."},
            8:  {"temp_min":20,"temp_max":32,"pluie":20,"floraison":["Caroube","Arbousier","Bruyère"],"activite":55,"conseil":"Reprise après été. Préparation hivernage. Traitement varroa si > 2%."},
            9:  {"temp_min":15,"temp_max":26,"pluie":40,"floraison":["Caroube","Bruyère","Arbousier"],"activite":50,"conseil":"Préparation hiver. Réduire espace intérieur. Assurer provisions 15 kg min."},
            10: {"temp_min":10,"temp_max":19,"pluie":65,"floraison":["Arbousier","Bruyère","Eucalyptus"],"activite":30,"conseil":"Hivernage précoce. Dernière inspection avant hiver. Traitement prophylactique."},
            11: {"temp_min":7,"temp_max":15,"pluie":70,"floraison":["Arbousier (début)","Eucalyptus"],"activite":15,"conseil":"Hivernage complet. Éviter ouvertures. Peser les ruches mensuellement."},
        }

        m = calendrier[idx_mois]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ Temp. min", f"{m['temp_min']}°C")
        col2.metric("🌡️ Temp. max", f"{m['temp_max']}°C")
        col3.metric("🌧️ Pluviométrie", f"{m['pluie']} mm")
        col4.metric("🐝 Activité butineuses", f"{m['activite']}%")

        st.markdown(f"""
        <div style='background:#0D2A1F;border:1px solid #1A5C3A;border-radius:8px;padding:14px;margin:12px 0'>
            <div style='font-size:.85rem;font-weight:600;color:#34D399;margin-bottom:6px'>🌸 Floraisons en cours — {mois_selected}</div>
            <div style='color:#F0F4FF;font-size:.85rem'>{' · '.join(m['floraison'])}</div>
        </div>
        <div style='background:#1E2535;border:1px solid #C8820A;border-radius:8px;padding:14px;margin:8px 0'>
            <div style='font-size:.85rem;font-weight:600;color:#F5A623;margin-bottom:4px'>💡 Conseil du mois</div>
            <div style='color:#F0F4FF;font-size:.85rem'>{m['conseil']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Graphique activité annuelle
        st.markdown("### 📈 Activité butineuses — Profil annuel Tlemcen")
        df_cal = pd.DataFrame({
            "Mois": mois_labels,
            "Activité (%)": [calendrier[i]["activite"] for i in range(12)],
            "Temp max (°C)": [calendrier[i]["temp_max"] for i in range(12)],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_cal["Mois"], y=df_cal["Activité (%)"],
                             name="Activité butineuses %", marker_color="#C8820A",
                             opacity=0.8))
        fig.add_trace(go.Scatter(x=df_cal["Mois"], y=df_cal["Temp max (°C)"],
                                  name="Temp max °C", line=dict(color="#F87171",width=2),
                                  yaxis="y2"))
        fig.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10,b=10,l=0,r=0),
            yaxis=dict(title="Activité %", range=[0,100]),
            yaxis2=dict(title="°C", overlaying="y", side="right", range=[0,45]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 🌡️ Saisie météo manuelle")
        conn = get_db()
        with st.form("meteo_form"):
            col1, col2, col3 = st.columns(3)
            date_meteo = col1.date_input("Date", datetime.date.today())
            temp_min   = col2.number_input("Temp min (°C)", -5.0, 50.0, 12.0, 0.5)
            temp_max   = col3.number_input("Temp max (°C)", -5.0, 55.0, 24.0, 0.5)
            col4, col5, col6 = st.columns(3)
            pluie      = col4.number_input("Pluie (mm)", 0.0, 200.0, 0.0, 1.0)
            vent       = col5.selectbox("Vent dominant", ["Calme","Brise légère","Vent modéré","Vent fort","Sirocco"])
            conditions = col6.selectbox("Conditions", ["Ensoleillé","Nuageux","Pluvieux","Brumeux","Orageux"])
            notes_m    = st.text_area("Observations terrain", height=60)
            submitted_m = st.form_submit_button("💾 Sauvegarder")

        if submitted_m:
            log_action("Météo saisie", f"{date_meteo}: {temp_min}-{temp_max}°C, {pluie}mm, {conditions}")
            st.success(f"✅ Météo du {date_meteo} enregistrée dans le journal.")
        conn.close()

    with tab3:
        st.markdown("### 🤖 Prédiction saisonnière IA personnalisée")
        if ia_active:
            mois_pred = st.multiselect("Mois à analyser", mois_labels,
                                        default=[mois_labels[datetime.date.today().month - 1]])
            if st.button("🤖 Générer le plan saisonnier IA", use_container_width=True):
                prompt = f"""Tu es expert apicole et agronome spécialisé dans la région de Tlemcen (Nord-Ouest Algérie, altitude 800m, climat méditerranéen semi-aride, zone bioclimatique sub-humide).

Pour les mois suivants : {', '.join(mois_pred)}

Génère un plan apicole mensuel ultra-détaillé avec :

## Pour chaque mois sélectionné :

### 🌸 1. Calendrier de floraison Tlemcen
- Espèces principales (nom latin + vernaculaire algérien)
- Durée de floraison et pic
- Source nectar/pollen/résine + qualité estimée

### 🐝 2. Activité de la colonie
- Stade de développement (couvain, population)
- Besoins nutritionnels spécifiques
- Risques pathologiques saisonniers (nosema, varroa, loque)

### 🍯 3. Prévisions de production
- Miel : type floral probable, rendement kg/ruche attendu
- Pollen : couleur, richesse protéique
- Propolis : disponibilité

### 🔧 4. Interventions prioritaires (liste numérotée)
Actions précises avec timing idéal (matin/soir, température recommandée)

### 💡 5. Astuce pro du mois
Un conseil exclusif adapté au rucher de l'Atlas algérien

Utilise des données chiffrées précises. Intègre les spécificités locales (sirocco estival, gel printanier tardif, miellée jujubier d'exception)."""

                with st.spinner("🤖 Génération du plan saisonnier..."):
                    result = ia_call(prompt)
                if result and not result.startswith("❌"):
                    afficher_resultat_ia(result, f"Plan saisonnier IA — {', '.join(mois_pred)}")
                    log_action("Météo IA prédiction", f"Mois : {', '.join(mois_pred)}")
                elif result:
                    st.error(result)
        else:
            st.info("🔑 Configurez une clé API pour activer les prédictions saisonnières IA.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE : ALERTES v3.0 — Système intelligent
# ════════════════════════════════════════════════════════════════════════════
def page_alertes():
    """Version améliorée avec système d'alertes intelligent et configurable."""
    st.markdown("## ⚠️ Alertes Intelligentes")
    st.markdown("<p style='color:#A8B4CC;margin-top:-10px'>Détection automatique · Score de risque · Actions recommandées</p>", unsafe_allow_html=True)

    conn = get_db()
    today = datetime.date.today()

    # Calculer les alertes dynamiques
    alertes = []

    # 1. Varroa critique
    df_varroa = pd.read_sql("""
        SELECT r.nom, r.id as ruche_id, i.varroa_pct, i.date_inspection
        FROM inspections i JOIN ruches r ON r.id=i.ruche_id
        WHERE i.date_inspection >= date('now','-7 days') AND i.varroa_pct >= 2.0
        ORDER BY i.varroa_pct DESC
    """, conn)
    for _, row in df_varroa.iterrows():
        niveau = "critique" if row["varroa_pct"] >= 3.0 else "attention"
        alertes.append({
            "niveau": niveau,
            "icone": "🔴" if niveau == "critique" else "🟡",
            "titre": f"Varroa {niveau.upper()} — {row['nom']}",
            "detail": f"Varroa à {row['varroa_pct']:.1f}% le {row['date_inspection']}",
            "action": "Traiter immédiatement à l'acide oxalique (T° < 10°C)" if niveau == "critique" else "Planifier traitement sous 7 jours",
            "ruche": row["nom"],
            "score_risque": min(100, int(row["varroa_pct"] * 25))
        })

    # 2. Ruches sans inspection récente
    df_retard = pd.read_sql("""
        SELECT r.nom,
               COALESCE(MAX(i.date_inspection), r.date_installation) as derniere_insp
        FROM ruches r LEFT JOIN inspections i ON i.ruche_id=r.id
        WHERE r.statut='actif'
        GROUP BY r.id, r.nom
        HAVING derniere_insp < date('now','-21 days') OR derniere_insp IS NULL
    """, conn)
    for _, row in df_retard.iterrows():
        alertes.append({
            "niveau": "attention",
            "icone": "🟡",
            "titre": f"Inspection en retard — {row['nom']}",
            "detail": f"Dernière inspection : {row['derniere_insp']}",
            "action": "Inspecter dès que possible — cadres, reine, varroa",
            "ruche": row["nom"],
            "score_risque": 45
        })

    # 3. Poids faible
    df_poids = pd.read_sql("""
        SELECT r.nom, i.poids_kg, i.date_inspection
        FROM inspections i JOIN ruches r ON r.id=i.ruche_id
        WHERE i.date_inspection >= date('now','-7 days') AND i.poids_kg < 15
        ORDER BY i.poids_kg ASC
    """, conn)
    for _, row in df_poids.iterrows():
        alertes.append({
            "niveau": "attention",
            "icone": "🟡",
            "titre": f"Poids faible — {row['nom']}",
            "detail": f"Poids : {row['poids_kg']} kg le {row['date_inspection']} (seuil : 15 kg)",
            "action": "Nourrir avec sirop 50/50 ou candi — vérifier reserves",
            "ruche": row["nom"],
            "score_risque": int(max(0, 60 - row["poids_kg"] * 3))
        })

    # 4. Traitements en cours terminant bientôt
    df_trait = pd.read_sql("""
        SELECT r.nom, t.produit, t.date_fin
        FROM traitements t JOIN ruches r ON r.id=t.ruche_id
        WHERE t.statut='en_cours' AND t.date_fin IS NOT NULL
        AND date(t.date_fin) BETWEEN date('now') AND date('now','+3 days')
    """, conn)
    for _, row in df_trait.iterrows():
        alertes.append({
            "niveau": "info",
            "icone": "🔵",
            "titre": f"Traitement se termine — {row['nom']}",
            "detail": f"{row['produit']} — fin prévue : {row['date_fin']}",
            "action": "Vérifier l'efficacité du traitement et noter résultats",
            "ruche": row["nom"],
            "score_risque": 20
        })

    # 5. Bonnes nouvelles — candidates élevage
    df_gr = pd.read_sql("""
        SELECT r.nom, SUM(rec.quantite_kg) as total, MAX(rec.hda_pct) as hda
        FROM recoltes rec JOIN ruches r ON r.id=rec.ruche_id
        WHERE rec.type_produit='gelée royale' GROUP BY r.nom HAVING total > 0.3
    """, conn)
    for _, row in df_gr.iterrows():
        alertes.append({
            "niveau": "success",
            "icone": "🟢",
            "titre": f"Excellente productrice — {row['nom']}",
            "detail": f"{row['total']:.2f} kg gelée royale{f' · 10-HDA {row[chr(104)+chr(100)+chr(97)]:.1f}%' if row['hda'] else ''}",
            "action": "Candidate idéale pour programme d'élevage sélectif",
            "ruche": row["nom"],
            "score_risque": 0
        })

    # Score de risque global
    if alertes:
        score_global = min(100, sum(a["score_risque"] for a in alertes if a["niveau"] in ["critique","attention"]) // max(1, len([a for a in alertes if a["niveau"] in ["critique","attention"]])))
        risk_color = "#F87171" if score_global >= 70 else ("#FBD147" if score_global >= 40 else "#34D399")
        st.markdown(f"""
        <div style='background:#1E2535;border:2px solid {risk_color};border-radius:12px;
                    padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:16px'>
            <div style='font-size:2rem'>{'🚨' if score_global >= 70 else ('⚠️' if score_global >= 40 else '✅')}</div>
            <div>
                <div style='font-size:1.1rem;font-weight:700;color:{risk_color}'>
                    Score de risque global : {score_global}/100
                </div>
                <div style='font-size:.82rem;color:#A8B4CC'>
                    {'État critique — Interventions immédiates requises' if score_global >= 70
                     else ('État préoccupant — Surveillance renforcée' if score_global >= 40
                     else 'État satisfaisant — Surveillance normale')}
                    · {len(alertes)} alertes actives · {len([a for a in alertes if a['niveau']=='critique'])} critiques
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Afficher les alertes groupées par niveau
    niveaux_ordre = [("critique","🔴 Alertes Critiques","#F87171"),
                     ("attention","🟡 Alertes Attention","#FBD147"),
                     ("info","🔵 Informations","#60A5FA"),
                     ("success","🟢 Points Positifs","#34D399")]

    for niveau, label, color in niveaux_ordre:
        alertes_niveau = [a for a in alertes if a["niveau"] == niveau]
        if alertes_niveau:
            st.markdown(f"### {label}")
            for a in alertes_niveau:
                bg = {"critique":"#2A0D0D","attention":"#2A200A","info":"#0D1A2A","success":"#0D2A1F"}[niveau]
                border = {"critique":"#5C1A1A","attention":"#4A3A10","info":"#1A3A5C","success":"#1A5C3A"}[niveau]

                st.markdown(f"""
                <div style='background:{bg};border:1px solid {border};border-left:4px solid {color};
                            border-radius:8px;padding:14px 16px;margin-bottom:10px'>
                    <div style='font-weight:600;color:{color};font-size:.95rem'>{a['icone']} {a['titre']}</div>
                    <div style='color:#A8B4CC;font-size:.82rem;margin:4px 0'>{a['detail']}</div>
                    <div style='color:#F0F4FF;font-size:.82rem'>💡 Action : {a['action']}</div>
                    {'<div style="color:#6B7A99;font-size:.72rem;margin-top:4px">Score risque : ' + str(a['score_risque']) + '/100</div>' if a.get('score_risque',0) > 0 else ''}
                </div>
                """, unsafe_allow_html=True)

    if not alertes:
        st.success("✅ Aucune alerte active — Rucher en parfait état !")

    # Analyse IA des alertes
    if alertes and ia_active:
        st.markdown("---")
        ia_active2 = get_api_key_for_provider(get_active_provider())
        if ia_active2 and st.button("🤖 Obtenir un plan d'action IA global", use_container_width=True):
            alertes_texte = "\n".join([f"- [{a['niveau'].upper()}] {a['titre']} : {a['detail']}" for a in alertes])
            prompt = f"""Tu es vétérinaire apicole et expert rucher nord-africain. Voici l'état d'alerte du rucher aujourd'hui :

{alertes_texte}

Génère un plan d'action prioritaire en français pour les 7 prochains jours :

## 🚨 1. Actions IMMÉDIATES (24-48h)
Pour chaque alerte critique, protocole exact d'intervention

## 📅 2. Planning semaine (J1 à J7)
Tableau des interventions recommandées avec timing optimal

## 🛡️ 3. Prévention
Actions préventives pour éviter la récurrence de ces problèmes

## 📊 4. Suivi recommandé
Paramètres à surveiller et fréquence de contrôle

Donne des instructions précises et pratiques pour un apiculteur algérien."""

            with st.spinner("🤖 Génération du plan d'action..."):
                result = ia_call(prompt)
            if result and not result.startswith("❌"):
                afficher_resultat_ia(result, "Plan d'action global — IA Vétérinaire Apicole")
                log_action("Plan IA alertes", f"{len(alertes)} alertes analysées")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PATCH : Remplacer init_db pour inclure v3
# ════════════════════════════════════════════════════════════════════════════
_original_main = main

def main():
    inject_css()
    init_db()
    init_db_v3()   # ← nouvelles tables

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
        return

    sidebar_v3()

    page = st.session_state.get("page", "dashboard")
    router = {
        "dashboard":    page_dashboard,
        "ruches":       page_ruches,
        "inspections":  page_inspections,
        "traitements":  page_traitements,
        "productions":  page_productions,
        "morpho":       page_morpho,
        "carto":        page_carto,
        "meteo":        page_meteo,
        "genetique":    page_genetique,
        "flore":        page_flore,
        "alertes":      page_alertes,
        "journal":      page_journal,
        "admin":        page_admin,
        # ── Nouvelles pages v3.0 ──────────────────────────
        "comptabilite": page_comptabilite,
        "agenda":       page_agenda,
        "miel":         page_analyseur_miel,
    }
    fn = router.get(page, page_dashboard)
    fn()

    st.markdown("""
    <div class='api-footer'>
        🐝 ApiTrack Pro v3.0 ULTIMATE · Streamlit + Python + SQLite + IA Multi-fournisseurs · Rucher de l'Atlas · 2025
        <br><span style='font-size:.65rem;color:#6B7A99'>Unique au monde — Comptabilité · Agenda IA · Analyseur Miel · Alertes Intelligentes · Météo Prédictive</span>
    </div>
    """, unsafe_allow_html=True)


def sidebar_v3():
    """Sidebar améliorée avec les nouvelles pages v3."""
    with st.sidebar:
        st.markdown("""
        <div style='padding:8px 0 16px;border-bottom:1px solid #3d2a0e;margin-bottom:12px'>
            <div style='font-size:1.6rem;margin-bottom:4px'>🐝</div>
            <div style='font-family:Playfair Display,serif;color:#F5A623;font-size:1.1rem;font-weight:600'>ApiTrack Pro</div>
            <div style='font-size:.65rem;color:#8899BB;text-transform:uppercase;letter-spacing:.1em'>v3.0 ULTIMATE</div>
        </div>
        """, unsafe_allow_html=True)

        rucher_nom = get_setting("rucher_nom", "Mon Rucher")
        st.markdown(f"<div style='font-size:.75rem;color:#6B7A99;margin-bottom:12px'>📍 {rucher_nom}</div>",
                    unsafe_allow_html=True)

        pages = {
            "🏠 Dashboard":          "dashboard",
            "🐝 Mes ruches":          "ruches",
            "🔍 Inspections":         "inspections",
            "💊 Traitements":         "traitements",
            "🍯 Productions":         "productions",
            "🧬 Morphométrie IA":     "morpho",
            "🗺️ Cartographie":        "carto",
            "☀️ Météo & Calendrier":  "meteo",
            "📊 Génétique":           "genetique",
            "🌿 Flore mellifère":     "flore",
            "⚠️ Alertes IA":          "alertes",
            "📋 Journal":             "journal",
            "─────────────────":     None,
            "💰 Comptabilité":        "comptabilite",
            "📆 Agenda & Tâches":     "agenda",
            "🔬 Analyseur Miel IA":   "miel",
            "─────────────────":     None,
            "⚙️ Administration":      "admin",
        }

        if "page" not in st.session_state:
            st.session_state.page = "dashboard"

        for label, key in pages.items():
            if key is None:
                st.sidebar.markdown(f"<div style='color:#3A4A66;font-size:.65rem;padding:2px 12px'>{label}</div>",
                                    unsafe_allow_html=True)
                continue
            is_new = key in ["comptabilite","agenda","miel"]
            label_display = f"{label} {'🆕' if is_new else ''}"
            if st.sidebar.button(label_display, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.sidebar.markdown("<hr style='border-color:#2E3A52;margin:12px 0'>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size:.75rem;color:#6B7A99'>👤 {st.session_state.get('username','admin')}</div>",
                            unsafe_allow_html=True)
        if st.sidebar.button("🚪 Déconnexion", use_container_width=True):
            log_action("Déconnexion", f"Utilisateur {st.session_state.get('username')} déconnecté")
            st.session_state.logged_in = False
            st.rerun()


# ── Surcharge du __main__ ──────────────────────────────────────────────────
