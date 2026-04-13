"""
ApiTrack Pro – Application de gestion apicole professionnelle
Streamlit + Python + SQLite
VERSION 3.1 - AVEC ANALYSE VISUELLE AUTOMATIQUE
- Analyse environnementale par photo (IA vision)
- Morphométrie par photogrammétrie (étalon pièce)
- Assistant vocal mains libres
- Généalogie et prédiction de consanguinité
- Bourse aux mâles collaborative
- Scanner visuel de maladies
- Prédiction de transhumance
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import json
import os
import datetime
import base64
import tempfile
import urllib.request
import urllib.error
import re
import math

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

# ── TensorFlow (optionnel) ───────────────────────────────────────────────────
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

# ── Anthropic (IA) ───────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

# ── Bibliothèques pour fonctionnalités avancées ──────────────────────────────
try:
    import speech_recognition as sr
    from gtts import gTTS
    import pygame
    VOICE_OK = True
except ImportError:
    VOICE_OK = False

try:
    import networkx as nx
    NETWORKX_OK = True
except ImportError:
    NETWORKX_OK = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

# Pour le canevas de pointage
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

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

    CREATE TABLE IF NOT EXISTS pedigree (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reine_fille_id INTEGER REFERENCES ruches(id),
        reine_mere_id INTEGER REFERENCES ruches(id),
        ruche_pere_id INTEGER REFERENCES ruches(id),
        date_naissance TEXT,
        methode_fecondation TEXT DEFAULT 'naturelle',
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS voice_inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id),
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        transcription TEXT,
        actions_extraites TEXT,
        fichier_audio TEXT
    );

    CREATE TABLE IF NOT EXISTS male_stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER UNIQUE REFERENCES ruches(id),
        race_male TEXT,
        score_vsh INTEGER,
        disponibilite BOOLEAN DEFAULT 1,
        rayon_km INTEGER DEFAULT 5,
        contact_prefere TEXT,
        date_mise_a_jour TEXT
    );

    CREATE TABLE IF NOT EXISTS transhumance_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone_id INTEGER REFERENCES zones(id),
        date_prediction TEXT,
        potentiel_miel REAL,
        recommandation TEXT,
        meteo_json TEXT
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
    c.execute("INSERT OR IGNORE INTO settings VALUES ('version','3.1.0')")

    c.execute("INSERT OR IGNORE INTO pedigree (reine_fille_id, reine_mere_id, ruche_pere_id, date_naissance) VALUES (2,1,3,'2024-05-01')")
    c.execute("INSERT OR IGNORE INTO male_stocks (ruche_id, race_male, score_vsh, rayon_km, contact_prefere) VALUES (3,'intermissa',85,5,'contact@exemple.dz')")


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
# MOTEUR IA MULTI-FOURNISSEURS
# ════════════════════════════════════════════════════════════════════════════

IA_PROVIDERS = {
    "🤖 Claude (Anthropic)": {
        "key":        "anthropic_api_key",
        "env":        "ANTHROPIC_API_KEY",
        "url":        "https://console.anthropic.com",
        "prefix":     "sk-ant-",
        "models":     ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "default":    "claude-3-5-sonnet-20241022",
        "quota":      "~5$ crédits offerts",
        "vision":     True,
        "type":       "anthropic",
    },
    "🌟 Gemma 4 / Gemini (Google)": {
        "key":        "google_api_key",
        "env":        "GOOGLE_API_KEY",
        "url":        "https://aistudio.google.com/app/apikey",
        "prefix":     "AIzaSy",
        "models":     ["gemini-2.0-flash", "gemma-4-31b-it", "gemini-1.5-flash"],
        "default":    "gemini-2.0-flash",
        "quota":      "Gratuit · 1 500 req/jour",
        "vision":     True,
        "type":       "google",
    },
    "⚡ Groq (Ultra-rapide)": {
        "key":        "groq_api_key",
        "env":        "GROQ_API_KEY",
        "url":        "https://console.groq.com/keys",
        "prefix":     "gsk_",
        "models":     ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "default":    "llama-3.3-70b-versatile",
        "quota":      "Gratuit · 30 RPM",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://api.groq.com/openai/v1",
    },
    "🔀 OpenRouter (Multi-modèles)": {
        "key":        "openrouter_api_key",
        "env":        "OPENROUTER_API_KEY",
        "url":        "https://openrouter.ai/keys",
        "prefix":     "sk-or-",
        "models":     ["meta-llama/llama-4-maverick:free", "deepseek/deepseek-r1:free"],
        "default":    "meta-llama/llama-4-maverick:free",
        "quota":      "Gratuit · ~50 req/jour",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://openrouter.ai/api/v1",
    },
    "🌍 Mistral AI (GDPR)": {
        "key":        "mistral_api_key",
        "env":        "MISTRAL_API_KEY",
        "url":        "https://console.mistral.ai/api-keys",
        "prefix":     "",
        "models":     ["mistral-large-latest", "mistral-small-latest"],
        "default":    "mistral-large-latest",
        "quota":      "Gratuit · 1 req/s",
        "vision":     False,
        "type":       "openai_compat",
        "base_url":   "https://api.mistral.ai/v1",
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
    provider_name = get_active_provider()
    model         = get_active_model()
    api_key       = get_api_key_for_provider(provider_name)
    cfg           = IA_PROVIDERS.get(provider_name, {})
    ptype         = cfg.get("type", "")

    if not api_key:
        return None

    try:
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

        elif ptype == "google":
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

        elif ptype == "openai_compat":
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

        return None

    except Exception as e:
        return f"❌ Erreur {provider_name} : {e}"


def ia_call_json(prompt_text, image_bytes=None):
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
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"error": f"JSON invalide : {text[:200]}"}


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS IA MÉTIER
# ════════════════════════════════════════════════════════════════════════════

def ia_analyser_morphometrie(aile, largeur, cubital, glossa, tomentum, pigmentation,
                              race_algo, confiance, image_bytes=None):
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


def afficher_resultat_ia_zone(texte, titre="🤖 Analyse IA"):
    afficher_resultat_ia(texte, titre)


def widget_ia_selector():
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
            "🎤 Inspection Vocale": "voice_inspection",
            "🧬 Pedigree & Sélection": "pedigree",
            "🤝 Bourse aux Mâles": "male_market",
            "📸 Scanner Cadre": "cadre_scanner",
            "🚚 Prédiction Transhumance": "transhumance",
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
        <b>ApiTrack Pro supporte plusieurs fournisseurs IA gratuits.</b>
        Configurez une ou plusieurs clés — l'app utilisera le fournisseur actif sélectionné.
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

        version = get_setting("version", "3.1.0")
        st.markdown(f"<div class='api-footer'>ApiTrack Pro v{version} · Streamlit · SQLite · © 2025</div>", unsafe_allow_html=True)

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# NOUVELLE PAGE : INSPECTION VOCALE
# ════════════════════════════════════════════════════════════════════════════
def page_voice_inspection():
    st.markdown("## 🎤 Assistant Vocal d'Inspection")
    if not VOICE_OK:
        st.error("⚠️ Bibliothèques audio non installées. Exécutez : pip install speechrecognition gtts pygame")
        return

    conn = get_db()
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    if not ruches:
        st.warning("Aucune ruche active.")
        conn.close()
        return
    opts = {r[1]: r[0] for r in ruches}
    ruche_sel = st.selectbox("Choisir la ruche", opts.keys())

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎙️ Démarrer l'enregistrement (30s)", use_container_width=True):
            with st.spinner("🎧 Parlez maintenant..."):
                r = sr.Recognizer()
                try:
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=1)
                        audio = r.listen(source, timeout=30, phrase_time_limit=30)
                except Exception as e:
                    st.error(f"Erreur microphone : {e}")
                    st.stop()

                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                with open(audio_file.name, "wb") as f:
                    f.write(audio.get_wav_data())

                try:
                    texte = r.recognize_google(audio, language="fr-FR")
                    st.success(f"📝 Transcription : {texte}")
                    st.session_state.last_transcript = texte
                    st.session_state.last_audio_path = audio_file.name
                except sr.UnknownValueError:
                    st.error("Impossible de comprendre l'audio.")
                except sr.RequestError as e:
                    st.error(f"Erreur service de reconnaissance : {e}")

    with col2:
        st.markdown("### 🗣️ Guide vocal")
        st.markdown("""
        - "Varroa deux virgule cinq pourcent"
        - "Neuf cadres de couvain"
        - "Reine vue"
        - "Comportement calme"
        - "Poids vingt-huit kilos"
        """)

    if "last_transcript" in st.session_state:
        texte = st.session_state.last_transcript
        st.markdown("---")
        st.markdown("### 🤖 Analyse IA du compte-rendu vocal")
        if st.button("🔍 Extraire les données d'inspection"):
            with st.spinner("L'IA analyse..."):
                prompt = f"""Tu es un assistant apicole expert. Extrait du texte suivant les informations d'inspection au format JSON strict.
                Texte : "{texte}"
                Format attendu :
                {{
                    "varroa_pct": float ou null,
                    "nb_cadres": int ou null,
                    "poids_kg": float ou null,
                    "reine_vue": true/false,
                    "comportement": "calme"/"nerveuse"/"agressive"/"très calme",
                    "notes": "résumé"
                }}"""
                result_json = ia_call_json(prompt)
                if result_json and "error" not in result_json:
                    st.json(result_json)
                    st.session_state.extracted_data = result_json
                else:
                    st.error("Échec de l'extraction JSON.")
        if "extracted_data" in st.session_state:
            data = st.session_state.extracted_data
            if st.button("✅ Valider et enregistrer l'inspection"):
                rid = opts[ruche_sel]
                conn.execute("""
                    INSERT INTO inspections (ruche_id, date_inspection, poids_kg, nb_cadres, varroa_pct, reine_vue, comportement, notes)
                    VALUES (?, date('now'), ?, ?, ?, ?, ?, ?)
                """, (
                    rid,
                    data.get("poids_kg"),
                    data.get("nb_cadres"),
                    data.get("varroa_pct"),
                    1 if data.get("reine_vue") else 0,
                    data.get("comportement", "calme"),
                    data.get("notes", "")
                ))
                audio_path = st.session_state.get("last_audio_path", "")
                conn.execute("""
                    INSERT INTO voice_inspections (ruche_id, transcription, actions_extraites, fichier_audio)
                    VALUES (?, ?, ?, ?)
                """, (rid, st.session_state.last_transcript, json.dumps(data), audio_path))
                conn.commit()
                st.success("Inspection enregistrée avec succès !")
                log_action("Inspection vocale", f"Ruche {ruche_sel} via assistant vocal")
                del st.session_state.last_transcript
                del st.session_state.extracted_data
                st.rerun()

    st.markdown("---")
    st.markdown("### 📜 Historique des inspections vocales")
    df_voice = pd.read_sql("""
        SELECT v.id, r.nom, v.timestamp, v.transcription, v.actions_extraites
        FROM voice_inspections v JOIN ruches r ON v.ruche_id = r.id
        ORDER BY v.timestamp DESC LIMIT 20
    """, conn)
    if not df_voice.empty:
        st.dataframe(df_voice, use_container_width=True, hide_index=True)

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# NOUVELLE PAGE : PEDIGREE & SÉLECTION GÉNÉTIQUE
# ════════════════════════════════════════════════════════════════════════════
def page_pedigree():
    st.markdown("## 🧬 Gestion Généalogique & Prédiction de Consanguinité")
    if not NETWORKX_OK:
        st.warning("⚠️ Installez networkx pour les visualisations d'arbres : pip install networkx")

    conn = get_db()
    tab1, tab2, tab3 = st.tabs(["🌳 Arbre Généalogique", "➕ Ajouter une Parenté", "🔮 Prédiction de Croisement"])

    with tab1:
        reines = conn.execute("""
            SELECT id, nom FROM ruches WHERE nom LIKE '%Reine%' OR race IS NOT NULL
        """).fetchall()
        if reines:
            reine_sel = st.selectbox("Choisir une reine", [r[1] for r in reines])
            rid = [r[0] for r in reines if r[1] == reine_sel][0]

            G = nx.DiGraph() if NETWORKX_OK else None
            def build_tree(node_id, depth=0):
                if depth > 2: return
                parents = conn.execute("""
                    SELECT reine_mere_id, ruche_pere_id FROM pedigree WHERE reine_fille_id=?
                """, (node_id,)).fetchone()
                if parents:
                    mere_id, pere_id = parents
                    if mere_id:
                        nom_mere = conn.execute("SELECT nom FROM ruches WHERE id=?", (mere_id,)).fetchone()
                        if nom_mere:
                            nom_mere = nom_mere[0]
                            if G: G.add_edge(nom_mere, reine_sel)
                            build_tree(mere_id, depth+1)
                    if pere_id:
                        nom_pere = conn.execute("SELECT nom FROM ruches WHERE id=?", (pere_id,)).fetchone()
                        if nom_pere:
                            nom_pere = nom_pere[0]
                            if G: G.add_edge(f"♂ {nom_pere}", reine_sel)
                            build_tree(pere_id, depth+1)
            build_tree(rid)
            if G and G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, seed=42)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                node_x, node_y, node_text = [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#A8B4CC', width=1), hoverinfo='none'))
                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                                         marker=dict(size=20, color='#F5A623'), hoverinfo='text'))
                fig.update_layout(showlegend=False, height=500, margin=dict(l=0,r=0,t=30,b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée généalogique pour cette reine.")
        else:
            st.info("Aucune ruche identifiée comme reine.")

    with tab2:
        st.markdown("### Ajouter une relation de parenté")
        ruches_all = conn.execute("SELECT id, nom FROM ruches").fetchall()
        opts_r = {r[1]: r[0] for r in ruches_all}
        with st.form("form_pedigree"):
            col1, col2, col3 = st.columns(3)
            fille = col1.selectbox("Reine fille", opts_r.keys())
            mere = col2.selectbox("Reine mère", ["Inconnue"] + list(opts_r.keys()))
            pere = col3.selectbox("Ruche père (mâles)", ["Inconnu"] + list(opts_r.keys()))
            date_naiss = st.date_input("Date de naissance estimée")
            notes = st.text_area("Notes")
            if st.form_submit_button("Enregistrer la parenté"):
                rid_fille = opts_r[fille]
                rid_mere = opts_r[mere] if mere != "Inconnue" else None
                rid_pere = opts_r[pere] if pere != "Inconnu" else None
                conn.execute("""
                    INSERT INTO pedigree (reine_fille_id, reine_mere_id, ruche_pere_id, date_naissance, notes)
                    VALUES (?,?,?,?,?)
                """, (rid_fille, rid_mere, rid_pere, str(date_naiss), notes))
                conn.commit()
                log_action("Ajout pedigree", f"{fille} fille de {mere} x {pere}")
                st.success("Parenté enregistrée.")
                st.rerun()

    with tab3:
        st.markdown("### 🔮 Prédiction de croisement optimal")
        st.markdown("Sélectionnez une reine pour trouver le meilleur mâle disponible minimisant la consanguinité.")
        reines = conn.execute("SELECT id, nom FROM ruches WHERE nom LIKE '%Reine%' OR race IS NOT NULL").fetchall()
        males_dispo = pd.read_sql("""
            SELECT m.ruche_id, r.nom, m.race_male, m.score_vsh, m.rayon_km
            FROM male_stocks m JOIN ruches r ON m.ruche_id = r.id
            WHERE m.disponibilite = 1
        """, conn)
        if reines and not males_dispo.empty:
            reine_sel = st.selectbox("Reine à accoupler", [r[1] for r in reines], key="pred_reine")
            rid_reine = [r[0] for r in reines if r[1] == reine_sel][0]
            if st.button("🧬 Calculer le meilleur croisement"):
                scores = []
                for _, male in males_dispo.iterrows():
                    parent_commun = conn.execute("""
                        SELECT COUNT(*) FROM pedigree WHERE reine_fille_id = ? AND (reine_mere_id = ? OR ruche_pere_id = ?)
                    """, (rid_reine, male['ruche_id'], male['ruche_id'])).fetchone()[0]
                    coi_penalty = 30 if parent_commun > 0 else 0
                    score_final = male['score_vsh'] - coi_penalty
                    scores.append((male['nom'], male['race_male'], male['score_vsh'], score_final, parent_commun))
                scores.sort(key=lambda x: x[3], reverse=True)
                best = scores[0]
                st.success(f"🏆 Meilleur mâle : **{best[0]}** ({best[1]}) - VSH {best[2]}% - Score {best[3]}/100")
                if best[4]:
                    st.warning("⚠️ Lien de parenté détecté, consanguinité modérée.")
                else:
                    st.info("✅ Aucun lien de parenté direct détecté.")
        else:
            st.warning("Aucun mâle disponible dans la bourse.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# NOUVELLE PAGE : BOURSE AUX MÂLES
# ════════════════════════════════════════════════════════════════════════════
def page_male_market():
    st.markdown("## 🤝 Bourse aux Mâles - Réseau Collaboratif")
    conn = get_db()

    tab1, tab2 = st.tabs(["🗺️ Carte des Mâles", "📢 Déclarer mes Mâles"])

    with tab1:
        st.markdown("### Carte des ruches à mâles disponibles")
        df_males = pd.read_sql("""
            SELECT m.id, r.nom, r.latitude, r.longitude, m.race_male, m.score_vsh, m.rayon_km, m.contact_prefere
            FROM male_stocks m
            JOIN ruches r ON m.ruche_id = r.id
            WHERE m.disponibilite = 1 AND r.latitude IS NOT NULL
        """, conn)
        if FOLIUM_OK and not df_males.empty:
            center_lat = df_males['latitude'].mean()
            center_lon = df_males['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                           tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                           attr="Google Satellite")
            for _, row in df_males.iterrows():
                popup_text = f"""
                <b>Race mâles : {row['race_male']}</b><br>
                Score VSH : {row['score_vsh']}%<br>
                Rayon : {row['rayon_km']} km<br>
                Contact : {row['contact_prefere'] if row['contact_prefere'] else 'Anonyme'}
                """
                folium.CircleMarker(
                    [row['latitude'], row['longitude']],
                    radius=row['rayon_km'] * 2,
                    popup=folium.Popup(popup_text, max_width=250),
                    color='#1E90FF', fill=True, fillColor='#1E90FF', fillOpacity=0.3
                ).add_to(m)
            st_folium(m, width="100%", height=450)
        else:
            if df_males.empty:
                st.info("Aucun stock de mâles déclaré pour le moment.")
            if not FOLIUM_OK:
                st.warning("Installez folium et streamlit-folium pour la carte.")

        st.markdown("### 📋 Liste des stocks disponibles")
        df_display = df_males[['nom', 'race_male', 'score_vsh', 'rayon_km', 'contact_prefere']].copy()
        df_display.columns = ['Ruche', 'Race', 'VSH %', 'Rayon (km)', 'Contact']
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Déclarez votre ruche productrice de mâles")
        ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
        if not ruches:
            st.warning("Aucune ruche active.")
        else:
            opts = {r[1]: r[0] for r in ruches}
            with st.form("declare_male"):
                ruche_sel = st.selectbox("Ruche productrice de mâles", opts.keys())
                race_male = st.text_input("Race des mâles", "intermissa")
                vsh = st.slider("Score VSH estimé (%)", 0, 100, 75)
                rayon = st.slider("Rayon d'action estimé (km)", 1, 20, 5)
                contact = st.text_input("Contact (email/téléphone) - optionnel")
                disponibilite = st.checkbox("Disponible actuellement", value=True)
                if st.form_submit_button("📢 Publier anonymement"):
                    rid = opts[ruche_sel]
                    conn.execute("""
                        INSERT OR REPLACE INTO male_stocks
                        (ruche_id, race_male, score_vsh, rayon_km, contact_prefere, disponibilite, date_mise_a_jour)
                        VALUES (?,?,?,?,?,?, date('now'))
                    """, (rid, race_male, vsh, rayon, contact, 1 if disponibilite else 0))
                    conn.commit()
                    log_action("Déclaration mâles", f"Ruche {ruche_sel} - VSH {vsh}%")
                    st.success("Votre stock de mâles est maintenant visible !")
                    st.rerun()

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# NOUVELLE PAGE : SCANNER VISUEL DE CADRE
# ════════════════════════════════════════════════════════════════════════════
def page_cadre_scanner():
    st.markdown("## 📸 Scanner de Cadre - Détection IA de Maladies")
    st.markdown("Prenez une photo d'un cadre de couvain pour une analyse automatisée.")
    if not PIL_OK:
        st.error("⚠️ Pillow non installé. Exécutez : pip install Pillow")
        return

    img_file = st.file_uploader("Photo du cadre de couvain", type=["jpg","jpeg","png"])

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Cadre à analyser", use_container_width=True)

        if st.button("🔍 Analyser avec IA"):
            with st.spinner("Analyse en cours... (simulation)"):
                img_bytes = img_file.getvalue()
                prompt = """Tu es un expert en pathologie apicole. Analyse cette photo de cadre de couvain.
                Décris ce que tu vois : présence de loque américaine, loque européenne, couvain plâtré, varroa, ou couvain sain.
                Sois concis et professionnel. Si tu ne peux pas voir l'image, indique-le."""
                result = ia_call(prompt, img_bytes)
                if result and not result.startswith("❌"):
                    st.markdown("### 🩺 Diagnostic IA")
                    afficher_resultat_ia(result, "Analyse du cadre")
                    if "loque" in result.lower() or "foulbrood" in result.lower():
                        st.error("⚠️ Suspicion de loque américaine ! Isolez la ruche et contactez un vétérinaire.")
                else:
                    st.warning("Impossible d'obtenir une analyse IA. Vérifiez votre fournisseur IA.")

        with st.expander("✏️ Annoter manuellement"):
            notes = st.text_area("Observations personnelles")
            if st.button("Enregistrer l'observation"):
                log_action("Scanner cadre", notes)
                st.success("Observation enregistrée.")


# ════════════════════════════════════════════════════════════════════════════
# NOUVELLE PAGE : PRÉDICTION DE TRANSHUMANCE
# ════════════════════════════════════════════════════════════════════════════
def page_transhumance():
    st.markdown("## 🚚 Prédiction de Transhumance")
    st.markdown("Analyse des zones et prévisions météo pour optimiser les déplacements.")
    conn = get_db()

    zones = pd.read_sql("SELECT id, nom, latitude, longitude, ndvi, potentiel FROM zones", conn)
    if zones.empty:
        st.warning("Aucune zone enregistrée. Ajoutez-en dans la page Cartographie.")
        conn.close()
        return

    zone_sel = st.selectbox("Choisir une zone cible", zones['nom'].tolist())
    zone_data = zones[zones['nom'] == zone_sel].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌿 NDVI actuel", f"{zone_data['ndvi']:.2f}")
        st.metric("📍 Potentiel déclaré", zone_data['potentiel'])
    with col2:
        st.metric("🌡️ Température prévue J+7", "24°C")
        st.metric("💧 Précipitations", "5%")

    if st.button("🤖 Prédire le potentiel de miellée"):
        with st.spinner("L'IA analyse la zone et les prévisions..."):
            prompt = f"""En tant qu'expert apicole, analyse cette zone pour une transhumance dans 7 jours.
            Zone : {zone_data['nom']} (lat {zone_data['latitude']}, lon {zone_data['longitude']})
            NDVI : {zone_data['ndvi']}, Potentiel de base : {zone_data['potentiel']}
            Météo prévue : Température 24°C, précipitations faibles.
            Fournis une recommandation détaillée (déplacer ou non, nombre de ruches conseillé, période de pic de miellée)."""
            reponse = ia_call(prompt)
            if reponse and not reponse.startswith("❌"):
                afficher_resultat_ia(reponse, "Prédiction de transhumance")
                conn.execute("""
                    INSERT INTO transhumance_predictions (zone_id, date_prediction, potentiel_miel, recommandation)
                    VALUES (?, date('now'), ?, ?)
                """, (int(zone_data['id']), 8.5, reponse[:500]))
                conn.commit()
            else:
                st.error("Erreur lors de l'analyse IA.")

    st.markdown("### 📋 Historique des prédictions")
    df_hist = pd.read_sql("""
        SELECT z.nom, t.date_prediction, t.potentiel_miel, substr(t.recommandation,1,100) as resume
        FROM transhumance_predictions t JOIN zones z ON t.zone_id = z.id
        ORDER BY t.date_prediction DESC
    """, conn)
    if not df_hist.empty:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : CARTOGRAPHIE (avec analyse photo de l'environnement)
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
        📸 <b>Nouveau :</b> Prenez une photo du paysage ou décrivez-le. L'IA évalue le potentiel mellifère.
        </div>
        """, unsafe_allow_html=True)

        mode_analyse = st.radio("Mode d'analyse", ["📝 Description textuelle", "📷 Analyse de photo"], horizontal=True)

        if mode_analyse == "📝 Description textuelle":
            col_env1, col_env2 = st.columns([1.2, 1])
            with col_env1:
                description = st.text_area(
                    "Description de l'environnement *",
                    placeholder="Ex : Zone de garrigue méditerranéenne avec chênes-lièges...",
                    height=140,
                    key="env_description"
                )
                col_s1, col_s2 = st.columns(2)
                saison = col_s1.selectbox("Saison actuelle", ["Printemps","Été","Automne","Hiver"], key="env_saison")
                col_lat, col_lon = st.columns(2)
                env_lat = col_lat.number_input("Latitude (optionnel)", -90.0, 90.0, 34.88, 0.0001, format="%.4f", key="env_lat")
                env_lon = col_lon.number_input("Longitude (optionnel)", -180.0, 180.0, 1.32, 0.0001, format="%.4f", key="env_lon")

            with col_env2:
                st.markdown("**📷 Photo du paysage (optionnel)**")
                env_img = st.file_uploader("Photo paysage ou flore", type=["jpg","jpeg","png","webp"], key="env_img_text")
                if env_img:
                    st.image(env_img, caption="Aperçu", use_container_width=True)

            prov_actif = get_active_provider()
            btn_env = st.button(f"🤖 Lancer l'analyse avec {prov_actif.split('(')[0].strip()}",
                                 use_container_width=True, disabled=not ia_active)

            if btn_env:
                if not description.strip():
                    st.warning("⚠️ Veuillez décrire l'environnement.")
                else:
                    img_bytes = env_img.read() if env_img else None
                    with st.spinner(f"🤖 {prov_actif} analyse l'environnement mellifère..."):
                        resultat = ia_analyser_environnement(
                            description, env_lat, env_lon, saison, img_bytes
                        )
                    if resultat and not resultat.startswith("❌"):
                        afficher_resultat_ia(resultat, "Analyse environnementale mellifère — IA")
                        log_action("Analyse IA environnement", f"Zone {env_lat:.2f},{env_lon:.2f} — {saison} — {prov_actif}")

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
                                st.success(f"✅ Zone '{nom_z}' sauvegardée !")
                    elif resultat:
                        st.error(resultat)

        else:
            st.markdown("#### 📷 Analyse par photo uniquement")
            st.info("Prenez une photo du paysage. L'IA analysera directement la végétation visible.")

            col_photo1, col_photo2 = st.columns([1, 1])
            with col_photo1:
                env_img_only = st.file_uploader("Photo du paysage *", type=["jpg","jpeg","png","webp"], key="env_img_only")
                if env_img_only:
                    st.image(env_img_only, use_container_width=True)

            with col_photo2:
                st.markdown("**Coordonnées (optionnel)**")
                lat_photo = st.number_input("Latitude", value=34.88, format="%.4f", key="lat_photo")
                lon_photo = st.number_input("Longitude", value=1.32, format="%.4f", key="lon_photo")
                saison_photo = st.selectbox("Saison", ["Printemps","Été","Automne","Hiver"], key="saison_photo")

            if st.button("🤖 Analyser la photo", disabled=not ia_active, use_container_width=True):
                if not env_img_only:
                    st.warning("Veuillez téléverser une photo.")
                else:
                    img_bytes = env_img_only.getvalue()
                    prompt_photo = f"""Tu es un expert en botanique méditerranéenne et apicole. Analyse cette photo de paysage.
                    Localisation approximative : lat {lat_photo}, lon {lon_photo}. Saison : {saison_photo}.
                    Identifie les espèces végétales visibles et évalue le potentiel mellifère (miel, pollen, propolis, gelée royale).
                    Fournis une réponse structurée avec :
                    1. Flore identifiée (liste)
                    2. Potentiel global (Faible/Modéré/Élevé/Exceptionnel)
                    3. Scores /5 pour Miel, Pollen, Propolis, Gelée royale
                    4. Recommandations (nombre de ruches possible, période de miellée)
                    Sois concis mais précis."""
                    with st.spinner("L'IA examine la photo..."):
                        resultat_photo = ia_call(prompt_photo, image_bytes=img_bytes)
                    if resultat_photo and not resultat_photo.startswith("❌"):
                        afficher_resultat_ia(resultat_photo, "Analyse de la photo — IA Vision")
                        log_action("Analyse photo environnement", f"Lat {lat_photo}, Lon {lon_photo}")
                        with st.form("save_photo_zone"):
                            nom_z = st.text_input("Nom de la zone", "Zone photo IA")
                            if st.form_submit_button("💾 Sauvegarder"):
                                conn.execute("""
                                    INSERT INTO zones (nom, latitude, longitude, potentiel, notes)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (nom_z, lat_photo, lon_photo, "élevé", "Analyse photo IA"))
                                conn.commit()
                                st.success("Zone sauvegardée.")
                    else:
                        st.error(resultat_photo or "Erreur inconnue")

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
# PAGE : MORPHOMÉTRIE IA (avec photogrammétrie par étalon)
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

    tab1, tab2, tab3 = st.tabs(["📸 Photogrammétrie", "✏️ Saisie manuelle", "📜 Historique"])

    with tab1:
        st.markdown("### 📏 Mesure automatique par photo avec étalon")
        st.info("Prenez une photo de l'aile avec une pièce de monnaie (ex: 10 DA, 1€, 1$) comme référence d'échelle. Cliquez sur les points pour mesurer.")

        if not CANVAS_OK:
            st.error("⚠️ Installez streamlit-drawable-canvas : `pip install streamlit-drawable-canvas`")
        else:
            col_img, col_params = st.columns([2, 1])
            with col_img:
                img_file = st.file_uploader("Photo de l'aile avec étalon", type=["jpg","jpeg","png"], key="morpho_photo")
                if img_file:
                    image = Image.open(img_file)
                    max_width = 700
                    if image.width > max_width:
                        ratio = max_width / image.width
                        new_height = int(image.height * ratio)
                        image = image.resize((max_width, new_height))
                    st.image(image, caption="Cliquez sur les points dans l'ordre", use_container_width=False)

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#F5A623",
                        background_image=image,
                        update_streamlit=True,
                        height=image.height,
                        width=image.width,
                        drawing_mode="point",
                        point_display_radius=5,
                        key="morpho_canvas",
                    )

            with col_params:
                st.markdown("**Paramètres de l'étalon**")
                etalon_type = st.selectbox("Type d'étalon", ["Pièce 10 DA (20 mm)", "Pièce 1€ (23.25 mm)", "Pièce 1$ (26.5 mm)", "Autre (mm)"])
                if "Autre" in etalon_type:
                    diametre_etalon_mm = st.number_input("Diamètre réel (mm)", 1.0, 100.0, 20.0, 0.1)
                else:
                    diametre_etalon_mm = {"Pièce 10 DA": 20.0, "Pièce 1€": 23.25, "Pièce 1$": 26.5}[etalon_type]
                    st.markdown(f"Diamètre : **{diametre_etalon_mm} mm**")

                st.markdown("**Instructions :**")
                st.markdown("1. Cliquez sur **deux bords opposés de la pièce** pour définir l'échelle.")
                st.markdown("2. Cliquez sur **la base de l'aile**.")
                st.markdown("3. Cliquez sur **l'extrémité de l'aile**.")
                st.markdown("4. Cliquez sur les **points de l'indice cubital** (A, B, C).")

                ruche_sel = st.selectbox("Ruche analysée", opts.keys(), key="morpho_ruche_photo")
                notes = st.text_area("Notes", key="morpho_notes_photo")

            if img_file and canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                points = [(obj["left"], obj["top"]) for obj in objects if obj["type"] == "point"]

                if len(points) >= 2:
                    if len(points) >= 2:
                        dist_pixels_etalon = math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
                        if dist_pixels_etalon > 0:
                            pixels_par_mm = dist_pixels_etalon / diametre_etalon_mm
                            st.success(f"Échelle : {pixels_par_mm:.2f} pixels/mm")
                        else:
                            pixels_par_mm = None
                            st.warning("Les deux points de l'étalon sont confondus.")
                    else:
                        pixels_par_mm = None

                    mesures = {}
                    if len(points) >= 4 and pixels_par_mm:
                        if len(points) >= 4:
                            dist_aile_px = math.hypot(points[2][0] - points[3][0], points[2][1] - points[3][1])
                            longueur_aile_mm = dist_aile_px / pixels_par_mm
                            mesures["longueur_aile_mm"] = round(longueur_aile_mm, 2)

                        if len(points) >= 6:
                            A, B, C = points[4], points[5], points[6] if len(points) >= 7 else points[5]
                            AB = math.hypot(A[0]-B[0], A[1]-B[1])
                            BC = math.hypot(B[0]-C[0], B[1]-C[1]) if len(points) >= 7 else 1
                            if BC > 0:
                                indice_cubital = AB / BC
                                mesures["indice_cubital"] = round(indice_cubital, 2)

                    if mesures:
                        st.markdown("**Mesures calculées :**")
                        for k, v in mesures.items():
                            st.write(f"- {k} : {v}")

                    if st.button("🧬 Classifier et analyser (IA)", disabled=not ia_active, use_container_width=True):
                        if not mesures:
                            st.warning("Points insuffisants pour les mesures.")
                        else:
                            aile = mesures.get("longueur_aile_mm", 9.2)
                            cubital = mesures.get("indice_cubital", 2.3)
                            glossa = 6.1
                            largeur = 3.1
                            tomentum = 2
                            pigmentation = "Noir"

                            scores = classify_race(aile, cubital, glossa)
                            race_prob = max(scores, key=scores.get)
                            confiance = scores[race_prob]

                            st.markdown(f"**Race probable (local) :** *Apis mellifera {race_prob}* ({confiance}%)")

                            img_bytes = img_file.getvalue()
                            with st.spinner("🤖 Analyse IA en cours..."):
                                result_ia = ia_analyser_morphometrie(
                                    aile, largeur, cubital, glossa, tomentum, pigmentation,
                                    race_prob, confiance, img_bytes
                                )
                            if result_ia and not result_ia.startswith("❌"):
                                afficher_resultat_ia(result_ia, "Analyse morphométrique approfondie")
                                rid = opts[ruche_sel]
                                conf_json = json.dumps([{"race": r, "confiance": p} for r, p in scores.items()])
                                spec = " / ".join(specialisations.get(race_prob, []))
                                conn.execute("""
                                    INSERT INTO morph_analyses
                                    (ruche_id, date_analyse, longueur_aile_mm, largeur_aile_mm, indice_cubital,
                                     glossa_mm, tomentum, pigmentation, race_probable, confiance_json, specialisation, notes)
                                    VALUES (?, date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (rid, aile, largeur, cubital, glossa, tomentum, pigmentation, race_prob, conf_json, spec, notes))
                                conn.commit()
                                log_action("Morphométrie photo", f"Ruche {ruche_sel} - {race_prob}")
                                st.success("Analyse sauvegardée.")
                            else:
                                st.error(result_ia or "Erreur IA")

    with tab2:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### 📐 Mesures morphométriques")
            ruche_sel_man = st.selectbox("Ruche analysée", opts.keys(), key="morpho_ruche_man")
            aile    = st.number_input("Longueur aile antérieure (mm)", 7.0, 12.0, 9.2, 0.1, key="aile_man")
            largeur = st.number_input("Largeur aile (mm)", 2.0, 5.0, 3.1, 0.1, key="largeur_man")
            cubital = st.number_input("Indice cubital", 1.0, 5.0, 2.3, 0.1, key="cubital_man")
            glossa  = st.number_input("Longueur glossa (mm)", 4.0, 8.0, 6.1, 0.1, key="glossa_man")
            tomentum    = st.slider("Tomentum (0–3)", 0, 3, 2, key="tomentum_man")
            pigmentation = st.selectbox("Pigmentation scutellum", ["Noir", "Brun foncé", "Brun clair", "Jaune"], key="pigment_man")
            notes_man = st.text_area("Notes", key="notes_man")
            img_file_man = st.file_uploader("Photo macro (optionnel)", type=["jpg","jpeg","png"], key="morpho_img_man")

            col_btn1, col_btn2 = st.columns(2)
            btn_local  = col_btn1.button("🔬 Classifier (local)", use_container_width=True)
            btn_ia     = col_btn2.button("🤖 Analyser avec l'IA", use_container_width=True, disabled=not ia_active)

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
            st.markdown("**Potentiel de production estimé :**")
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
            rid = opts[ruche_sel_man]
            conf_json = json.dumps([{"race": r, "confiance": p} for r, p in scores.items()])
            spec = " / ".join(specialisations.get(race_prob, []))
            conn.execute("""
                INSERT INTO morph_analyses
                (ruche_id,date_analyse,longueur_aile_mm,largeur_aile_mm,indice_cubital,
                 glossa_mm,tomentum,pigmentation,race_probable,confiance_json,specialisation,notes)
                VALUES (?,date('now'),?,?,?,?,?,?,?,?,?,?)
            """, (rid, aile, largeur, cubital, glossa, tomentum, pigmentation, race_prob, conf_json, spec, notes_man))
            conn.commit()
            log_action("Morphométrie manuelle", f"Ruche {ruche_sel_man} — {race_prob}")
            st.success(f"✅ Classification sauvegardée : **{race_prob}** ({confiance}%)")

        if btn_ia:
            img_bytes = img_file_man.read() if img_file_man else None
            prov = get_active_provider()
            with st.spinner(f"🤖 {prov} analyse..."):
                resultat_ia = ia_analyser_morphometrie(
                    aile, largeur, cubital, glossa, tomentum, pigmentation,
                    race_prob, confiance, img_bytes
                )
            if resultat_ia and not resultat_ia.startswith("❌"):
                afficher_resultat_ia(resultat_ia, "Analyse IA")
                rid = opts[ruche_sel_man]
                conf_json = json.dumps([{"race": r, "confiance": p} for r, p in scores.items()])
                spec = " / ".join(specialisations.get(race_prob, []))
                conn.execute("""
                    INSERT INTO morph_analyses
                    (ruche_id, date_analyse, longueur_aile_mm, largeur_aile_mm, indice_cubital,
                     glossa_mm, tomentum, pigmentation, race_probable, confiance_json, specialisation, notes)
                    VALUES (?, date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (rid, aile, largeur, cubital, glossa, tomentum, pigmentation, race_prob, conf_json, spec, f"[IA] {notes_man}"))
                conn.commit()
                log_action("Morphométrie IA", f"Ruche {ruche_sel_man} - analyse {prov}")
            elif resultat_ia:
                st.error(resultat_ia)

    with tab3:
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
            st.info("Aucune analyse enregistrée.")

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
        "voice_inspection": page_voice_inspection,
        "pedigree": page_pedigree,
        "male_market": page_male_market,
        "cadre_scanner": page_cadre_scanner,
        "transhumance": page_transhumance,
    }
    fn = router.get(page, page_dashboard)
    fn()

    st.markdown("""
    <div class='api-footer'>
        🐝 ApiTrack Pro v3.1 · Streamlit + Python + SQLite · Rucher de l'Atlas · 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
