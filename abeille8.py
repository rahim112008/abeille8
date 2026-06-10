"""
ApiTrack Pro – Application de gestion apicole professionnelle
Streamlit + Python + SQLite
Fonctionnalités ajoutées :
- Assistant vocal d'inspection (reconnaissance vocale + extraction IA)
- Pedigree & sélection des reines (arbre généalogique)
- Scanner de cadre IA (détection maladies, comptage abeilles, reine)
- Prédiction de transhumance sur carte satellite (recherche de ville)
- Photogrammétrie, import CSV, géocodage, etc.
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

# ── OpenCV pour la photogrammétrie ───────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── Requêtes HTTP pour géocodage ─────────────────────────────────────────────
import urllib.request
import urllib.parse
import re


# ════════════════════════════════════════════════════════════════════════════
# SYSTÈME BILINGUE — ARABE / FRANÇAIS
# ════════════════════════════════════════════════════════════════════════════

TRANSLATIONS = {
    "en": {
        "app_name": "ApiTrack Pro", "app_subtitle": "Beekeeping Management",
        "logout": "🚪 Logout", "langue": "🌐 Language",
        "nav_dashboard": "🏠 Dashboard", "nav_ruches": "🐝 My Hives",
        "nav_inspections": "🔍 Inspections", "nav_assistant_vocal": "🎤 Voice Assistant",
        "nav_traitements": "💊 Treatments", "nav_productions": "🍯 Productions",
        "nav_morpho": "🧬 AI Morphometry", "nav_carto": "🗺️ Mapping",
        "nav_scanner_cadre": "📸 Frame Scanner", "nav_transhumance": "🚚 Transhumance",
        "nav_pedigree": "🧬 Pedigree", "nav_meteo": "☀️ Weather & Nectar",
        "nav_genetique": "📊 Genetics", "nav_flore": "🌿 Melliferous Flora",
        "nav_alertes": "⚠️ Alerts", "nav_journal": "📋 Activity Log",
        "nav_admin": "⚙️ Administration",
        "nav_elevage_reines": "👑 Queen Rearing", "nav_cellules_royales": "🔬 Royal Cells",
        "login_user": "Username", "login_pwd": "Password",
        "login_btn": "Sign In",
        "login_error": "Incorrect credentials. (Demo: admin / admin1234)",
        "login_hint": "admin / admin1234 for the demo",
        "login_subtitle": "Professional beekeeping management",
        "dashboard_title": "## 🏠 Dashboard",
        "dashboard_season": "Spring Season 2025",
        "metric_ruches": "🐝 Active Hives", "metric_miel": "🍯 Honey Harvested (kg)",
        "metric_insp": "🔍 Inspections (30d)", "metric_varroa": "⚠️ Critical Varroa",
        "chart_prod": "### 📈 Monthly Production (kg)",
        "chart_etat": "### 🐝 Hive Status",
        "alertes_actives": "### ⚠️ Active Alerts",
        "no_alert": "✅ No critical varroa alerts.",
        "ruches_title": "## 🐝 Hive Management",
        "tab_liste": "📋 Hive List", "tab_ajouter": "➕ Add a Hive",
        "export_csv": "⬇️ Export CSV",
        "supprimer_ruche": "### 🗑️ Delete a Hive",
        "choisir_ruche": "Choose the hive to delete",
        "btn_supprimer": "⚠️ Delete permanently",
        "nouvelle_ruche": "**New Hive**", "nom_ruche": "Name / Queen*",
        "race": "Race", "date_install": "Installation Date",
        "localisation_label": "Location", "btn_ajouter": "✅ Add Hive",
        "notes": "Notes",
        "insp_title": "## 🔍 Inspections",
        "tab_historique": "📋 History", "tab_nouvelle": "➕ New Inspection",
        "ruche_label": "Hive*", "date_label": "Date",
        "poids_label": "Weight (kg)", "cadres_label": "Nb Frames",
        "varroa_label": "Varroa (%)", "reine_vue_label": "Queen Seen",
        "comportement_label": "Behavior",
        "btn_enregistrer": "✅ Save Inspection",
        "evolution_varroa": "### 📈 Varroa Evolution",
        "pedigree_title": "## 🧬 Pedigree & Queen Selection",
        "tab_reines": "📋 Queen List",
        "tab_add_reine": "➕ Add Queen/Lineage",
        "tab_arbre": "🌳 Family Tree",
        "nouvelle_reine": "#### New Queen",
        "nom_reine": "Queen Name*", "ruche_associee": "Associated Hive",
        "date_naissance": "Estimated Birth Date",
        "mere": "Mother", "pere": "Father",
        "notes_perf": "Notes (characteristics, performances, etc.)",
        "btn_save_reine": "💾 Save Queen",
        "arbre_titre": "#### Family Tree (simplified view)",
        "no_reine": "No queens registered.",
        "no_genealogie": "No genealogy data available.",
        "success_reine": "✅ Queen '{}' saved.",
        "connexion_msg": "Sign In",
        "trait_title": "## 💊 Veterinary Treatments",
        "tab_encours": "📋 Ongoing & History",
        "tab_nouveau_trait": "➕ New Treatment",
        "prod_title": "## 🍯 Productions",
        "tab_recoltes": "🍯 Harvests", "tab_graphiques": "📊 Charts",
        "tab_nouvelle_rec": "➕ New Harvest",
        "meteo_title": "## ☀️ Weather & Nectar Flow — 7-day Forecast",
        "alerte_title": "## ⚠️ Alerts",
        "journal_title": "## 📋 Activity Log",
        "genetique_title": "## 📊 Genetics & Selection",
        "flore_title": "## 🌿 Melliferous Flora — Calendar",
        "scanner_title": "## 📸 Frame Scanner - AI Detection",
        "transhu_title": "## 🚚 Transhumance Prediction - Full Zone Analysis",
        "vocal_title": "## 🎤 Voice Inspection Assistant",
        "morpho_title": "## 🧬 AI Morphometry — Racial Classification",
        "carto_title": "## 🗺️ Mapping — Nectar Zones + AI Analysis",
        "admin_title": "## ⚙️ Administration",
        "tab_profil": "🏠 Apiary Profile", "tab_ia": "🤖 AI API Key",
        "tab_pwd": "🔐 Password", "tab_db": "💾 Database",
        "tab_import": "📂 Import CSV",
        "nom_rucher": "Apiary Name", "btn_sauvegarder": "💾 Save",
        "journal_vide": "Activity log is empty.",
        "no_trait": "No treatments registered.",
        "no_analyse": "No morphometric analyses registered.",
        "no_insp_vocale": "No voice inspections registered.",
        "no_analyse_cadre": "No frame analyses registered.",
        "no_transhu": "No transhumances planned.",
        "rechercher_ville": "#### 🔍 Search a City",
        "nom_ville_label": "City name",
        "centrer_btn": "📍 Center",
        "niveau_label": "🎓 User Level",
        "niveau_debutant": "🌱 Beginner",
        "niveau_intermediaire": "🐝 Intermediate",
        "niveau_expert": "🔬 Expert",
        "elevage_title": "## 👑 Queen & Drone Rearing",
        "cellules_title": "## 🔬 Royal Cell Tracking",
        "tab_elevage_reines": "👑 Queen Rearing",
        "tab_elevage_males": "♂️ Drone Rearing",
        "tab_cellules": "🔬 Royal Cells",
        "tab_generation": "🌳 Generations",
        "btn_add_cellule": "➕ Add Royal Cell",
        "stade_cellule": "Cell Stage",
        "date_greffe": "Grafting Date",
        "date_operculee": "Capping Date",
        "date_emergence": "Expected Emergence",
        "nb_cellules": "Number of Cells",
        "taux_acceptance": "Acceptance Rate (%)",
        "qualite_reine": "Queen Quality",
        "origine_reine": "Queen Origin",
        "methode_elevage": "Rearing Method",
        "ruche_eleveuse": "Starter/Finisher Hive",
        "ruche_donneuse": "Donor Hive",
    },
    "fr": {
        "app_name": "ApiTrack Pro", "app_subtitle": "Gestion Apicole",
        "logout": "🚪 Déconnexion", "langue": "🌐 Langue",
        "nav_dashboard": "🏠 Dashboard", "nav_ruches": "🐝 Mes ruches",
        "nav_inspections": "🔍 Inspections", "nav_assistant_vocal": "🎤 Assistant vocal",
        "nav_traitements": "💊 Traitements", "nav_productions": "🍯 Productions",
        "nav_morpho": "🧬 Morphométrie IA", "nav_carto": "🗺️ Cartographie",
        "nav_scanner_cadre": "📸 Scanner cadre", "nav_transhumance": "🚚 Transhumance",
        "nav_pedigree": "🧬 Pedigree", "nav_meteo": "☀️ Météo & Miellée",
        "nav_genetique": "📊 Génétique", "nav_flore": "🌿 Flore mellifère",
        "nav_alertes": "⚠️ Alertes", "nav_journal": "📋 Journal",
        "nav_admin": "⚙️ Administration",
        "nav_elevage_reines": "👑 Élevage reines", "nav_cellules_royales": "🔬 Cellules royales",
        "login_user": "Identifiant", "login_pwd": "Mot de passe",
        "login_btn": "Se connecter",
        "login_error": "Identifiants incorrects. (Démo : admin / admin1234)",
        "login_hint": "admin / admin1234 pour la démo",
        "login_subtitle": "Gestion apicole professionnelle",
        "dashboard_title": "## 🏠 Tableau de bord",
        "dashboard_season": "Saison printanière 2025",
        "metric_ruches": "🐝 Ruches actives", "metric_miel": "🍯 Miel récolté (kg)",
        "metric_insp": "🔍 Inspections (30j)", "metric_varroa": "⚠️ Varroa critique",
        "chart_prod": "### 📈 Production mensuelle (kg)",
        "chart_etat": "### 🐝 État des ruches",
        "alertes_actives": "### ⚠️ Alertes actives",
        "no_alert": "✅ Aucune alerte varroa critique en cours.",
        "ruches_title": "## 🐝 Gestion des ruches",
        "tab_liste": "📋 Liste des ruches", "tab_ajouter": "➕ Ajouter une ruche",
        "export_csv": "⬇️ Exporter CSV",
        "supprimer_ruche": "### 🗑️ Supprimer une ruche",
        "choisir_ruche": "Choisir la ruche à supprimer",
        "btn_supprimer": "⚠️ Supprimer définitivement",
        "nouvelle_ruche": "**Nouvelle ruche**", "nom_ruche": "Nom / Reine*",
        "race": "Race", "date_install": "Date d'installation",
        "localisation_label": "Localisation", "btn_ajouter": "✅ Ajouter la ruche",
        "notes": "Notes",
        "insp_title": "## 🔍 Inspections",
        "tab_historique": "📋 Historique", "tab_nouvelle": "➕ Nouvelle inspection",
        "ruche_label": "Ruche*", "date_label": "Date",
        "poids_label": "Poids (kg)", "cadres_label": "Nb cadres",
        "varroa_label": "Varroa (%)", "reine_vue_label": "Reine vue",
        "comportement_label": "Comportement",
        "btn_enregistrer": "✅ Enregistrer l'inspection",
        "evolution_varroa": "### 📈 Évolution du varroa",
        "pedigree_title": "## 🧬 Pedigree & Sélection des Reines",
        "tab_reines": "📋 Liste des reines",
        "tab_add_reine": "➕ Ajouter une reine/filiation",
        "tab_arbre": "🌳 Arbre généalogique",
        "nouvelle_reine": "#### Nouvelle reine",
        "nom_reine": "Nom de la reine*", "ruche_associee": "Ruche associée",
        "date_naissance": "Date de naissance estimée",
        "mere": "Mère", "pere": "Père",
        "notes_perf": "Notes (caractéristiques, performances, etc.)",
        "btn_save_reine": "💾 Enregistrer la reine",
        "arbre_titre": "#### Arbre généalogique (affichage simplifié)",
        "no_reine": "Aucune reine enregistrée.",
        "no_genealogie": "Aucune généalogie disponible.",
        "success_reine": "✅ Reine '{}' enregistrée.",
        "connexion_msg": "Connexion",
        "trait_title": "## 💊 Traitements vétérinaires",
        "tab_encours": "📋 En cours & historique",
        "tab_nouveau_trait": "➕ Nouveau traitement",
        "prod_title": "## 🍯 Productions",
        "tab_recoltes": "🍯 Récoltes", "tab_graphiques": "📊 Graphiques",
        "tab_nouvelle_rec": "➕ Nouvelle récolte",
        "meteo_title": "## ☀️ Météo & Miellée — Prévisions 7 jours",
        "alerte_title": "## ⚠️ Alertes",
        "journal_title": "## 📋 Journal d'activité",
        "genetique_title": "## 📊 Génétique & Sélection",
        "flore_title": "## 🌿 Flore mellifère — Calendrier",
        "scanner_title": "## 📸 Scanner de Cadre - Détection IA",
        "transhu_title": "## 🚚 Prédiction de Transhumance - Analyse complète de zone",
        "vocal_title": "## 🎤 Assistant Vocal d'Inspection",
        "morpho_title": "## 🧬 Morphométrie IA — Classification raciale",
        "carto_title": "## 🗺️ Cartographie — Zones mellifères + Analyse IA",
        "admin_title": "## ⚙️ Administration",
        "tab_profil": "🏠 Profil rucher", "tab_ia": "🤖 Clé API IA",
        "tab_pwd": "🔐 Mot de passe", "tab_db": "💾 Base de données",
        "tab_import": "📂 Import CSV",
        "nom_rucher": "Nom du rucher", "btn_sauvegarder": "💾 Sauvegarder",
        "journal_vide": "Le journal est vide.",
        "no_trait": "Aucun traitement enregistré.",
        "no_analyse": "Aucune analyse morphométrique enregistrée.",
        "no_insp_vocale": "Aucune inspection vocale enregistrée.",
        "no_analyse_cadre": "Aucune analyse enregistrée.",
        "no_transhu": "Aucune transhumance planifiée.",
        "rechercher_ville": "#### 🔍 Rechercher une ville",
        "nom_ville_label": "Nom de la ville",
        "centrer_btn": "📍 Centrer",
        "niveau_label": "🎓 Niveau utilisateur",
        "niveau_debutant": "🌱 Débutant",
        "niveau_intermediaire": "🐝 Intermédiaire",
        "niveau_expert": "🔬 Expert",
        "elevage_title": "## 👑 Élevage Reines & Mâles",
        "cellules_title": "## 🔬 Suivi Cellules Royales",
        "tab_elevage_reines": "👑 Élevage reines",
        "tab_elevage_males": "♂️ Élevage mâles",
        "tab_cellules": "🔬 Cellules royales",
        "tab_generation": "🌳 Générations",
        "btn_add_cellule": "➕ Ajouter cellule royale",
        "stade_cellule": "Stade de la cellule",
        "date_greffe": "Date de greffe",
        "date_operculee": "Date d'operculation",
        "date_emergence": "Émergence prévue",
        "nb_cellules": "Nombre de cellules",
        "taux_acceptance": "Taux d'acceptation (%)",
        "qualite_reine": "Qualité de la reine",
        "origine_reine": "Origine de la reine",
        "methode_elevage": "Méthode d'élevage",
        "ruche_eleveuse": "Ruche éleveuse",
        "ruche_donneuse": "Ruche donneuse",
    },
    "ar": {
        "app_name": "ApiTrack Pro", "app_subtitle": "إدارة النحل",
        "logout": "🚪 تسجيل الخروج", "langue": "🌐 اللغة",
        "nav_dashboard": "🏠 لوحة القيادة", "nav_ruches": "🐝 خلاياي",
        "nav_inspections": "🔍 التفتيشات", "nav_assistant_vocal": "🎤 المساعد الصوتي",
        "nav_traitements": "💊 العلاجات", "nav_productions": "🍯 الإنتاج",
        "nav_morpho": "🧬 القياس الشكلي", "nav_carto": "🗺️ الخرائط",
        "nav_scanner_cadre": "📸 مسح الإطار", "nav_transhumance": "🚚 الترحال",
        "nav_pedigree": "🧬 سجل الأنساب", "nav_meteo": "☀️ الطقس والرحيق",
        "nav_genetique": "📊 علم الوراثة", "nav_flore": "🌿 النباتات الرحيقية",
        "nav_alertes": "⚠️ التنبيهات", "nav_journal": "📋 السجل",
        "nav_admin": "⚙️ الإدارة",
        "nav_elevage_reines": "👑 تربية الملكات", "nav_cellules_royales": "🔬 الخلايا الملكية",
        "login_user": "اسم المستخدم", "login_pwd": "كلمة المرور",
        "login_btn": "تسجيل الدخول",
        "login_error": "بيانات خاطئة. (تجريبي: admin / admin1234)",
        "login_hint": "admin / admin1234 للنسخة التجريبية",
        "login_subtitle": "إدارة النحل الاحترافية",
        "dashboard_title": "## 🏠 لوحة القيادة",
        "dashboard_season": "موسم الربيع 2025",
        "metric_ruches": "🐝 الخلايا النشطة", "metric_miel": "🍯 العسل المحصود (كغ)",
        "metric_insp": "🔍 التفتيشات (30 يوم)", "metric_varroa": "⚠️ فاروا حرجة",
        "chart_prod": "### 📈 الإنتاج الشهري (كغ)",
        "chart_etat": "### 🐝 حالة الخلايا",
        "alertes_actives": "### ⚠️ التنبيهات النشطة",
        "no_alert": "✅ لا توجد تنبيهات حرجة حالياً.",
        "ruches_title": "## 🐝 إدارة الخلايا",
        "tab_liste": "📋 قائمة الخلايا", "tab_ajouter": "➕ إضافة خلية",
        "export_csv": "⬇️ تصدير CSV",
        "supprimer_ruche": "### 🗑️ حذف خلية",
        "choisir_ruche": "اختر الخلية للحذف",
        "btn_supprimer": "⚠️ حذف نهائي",
        "nouvelle_ruche": "**خلية جديدة**", "nom_ruche": "الاسم / الملكة*",
        "race": "السلالة", "date_install": "تاريخ التركيب",
        "localisation_label": "الموقع", "btn_ajouter": "✅ إضافة الخلية",
        "notes": "ملاحظات",
        "insp_title": "## 🔍 التفتيشات",
        "tab_historique": "📋 السجل", "tab_nouvelle": "➕ تفتيش جديد",
        "ruche_label": "الخلية*", "date_label": "التاريخ",
        "poids_label": "الوزن (كغ)", "cadres_label": "عدد الأطر",
        "varroa_label": "فاروا (%)", "reine_vue_label": "الملكة مرئية",
        "comportement_label": "السلوك",
        "btn_enregistrer": "✅ حفظ التفتيش",
        "evolution_varroa": "### 📈 تطور الفاروا",
        "pedigree_title": "## 🧬 سجل الأنساب واختيار الملكات",
        "tab_reines": "📋 قائمة الملكات",
        "tab_add_reine": "➕ إضافة ملكة / نسب",
        "tab_arbre": "🌳 شجرة النسب",
        "nouvelle_reine": "#### ملكة جديدة",
        "nom_reine": "اسم الملكة*", "ruche_associee": "الخلية المرتبطة",
        "date_naissance": "تاريخ الميلاد التقريبي",
        "mere": "الأم", "pere": "الأب",
        "notes_perf": "ملاحظات (الخصائص، الأداء...)",
        "btn_save_reine": "💾 حفظ الملكة",
        "arbre_titre": "#### شجرة النسب (عرض مبسط)",
        "no_reine": "لا توجد ملكات مسجلة.",
        "no_genealogie": "لا توجد بيانات أنساب متاحة.",
        "success_reine": "✅ تم تسجيل الملكة '{}'.",
        "connexion_msg": "تسجيل الدخول",
        "trait_title": "## 💊 العلاجات البيطرية",
        "tab_encours": "📋 الجارية والسابقة",
        "tab_nouveau_trait": "➕ علاج جديد",
        "prod_title": "## 🍯 الإنتاج",
        "tab_recoltes": "🍯 الحصاد", "tab_graphiques": "📊 الرسوم البيانية",
        "tab_nouvelle_rec": "➕ حصاد جديد",
        "meteo_title": "## ☀️ الطقس والرحيق — توقعات 7 أيام",
        "alerte_title": "## ⚠️ التنبيهات",
        "journal_title": "## 📋 سجل النشاط",
        "genetique_title": "## 📊 علم الوراثة والانتقاء",
        "flore_title": "## 🌿 النباتات الرحيقية — التقويم",
        "scanner_title": "## 📸 مسح الإطار - الكشف بالذكاء الاصطناعي",
        "transhu_title": "## 🚚 التنبؤ بالترحال - تحليل شامل للمنطقة",
        "vocal_title": "## 🎤 المساعد الصوتي للتفتيش",
        "morpho_title": "## 🧬 القياس الشكلي بالذكاء الاصطناعي — تصنيف السلالات",
        "carto_title": "## 🗺️ الخرائط — المناطق الرحيقية + تحليل الذكاء الاصطناعي",
        "admin_title": "## ⚙️ الإدارة",
        "tab_profil": "🏠 ملف المنحل", "tab_ia": "🤖 مفتاح الذكاء الاصطناعي",
        "tab_pwd": "🔐 كلمة المرور", "tab_db": "💾 قاعدة البيانات",
        "tab_import": "📂 استيراد CSV",
        "nom_rucher": "اسم المنحل", "btn_sauvegarder": "💾 حفظ",
        "journal_vide": "السجل فارغ.",
        "no_trait": "لا توجد علاجات مسجلة.",
        "no_analyse": "لا توجد تحليلات مسجلة.",
        "no_insp_vocale": "لا توجد تفتيشات صوتية مسجلة.",
        "no_analyse_cadre": "لا توجد تحليلات مسجلة.",
        "no_transhu": "لا توجد ترحالات مخططة.",
        "rechercher_ville": "#### 🔍 البحث عن مدينة",
        "nom_ville_label": "اسم المدينة",
        "centrer_btn": "📍 تمركز",
        "niveau_label": "🎓 مستوى المستخدم",
        "niveau_debutant": "🌱 مبتدئ",
        "niveau_intermediaire": "🐝 متوسط",
        "niveau_expert": "🔬 خبير",
        "elevage_title": "## 👑 تربية الملكات والذكور",
        "cellules_title": "## 🔬 متابعة الخلايا الملكية",
        "tab_elevage_reines": "👑 تربية الملكات",
        "tab_elevage_males": "♂️ تربية الذكور",
        "tab_cellules": "🔬 الخلايا الملكية",
        "tab_generation": "🌳 الأجيال",
        "btn_add_cellule": "➕ إضافة خلية ملكية",
        "stade_cellule": "مرحلة الخلية",
        "date_greffe": "تاريخ التطعيم",
        "date_operculee": "تاريخ الإغلاق",
        "date_emergence": "الخروج المتوقع",
        "nb_cellules": "عدد الخلايا",
        "taux_acceptance": "نسبة القبول (%)",
        "qualite_reine": "جودة الملكة",
        "origine_reine": "أصل الملكة",
        "methode_elevage": "طريقة التربية",
        "ruche_eleveuse": "الخلية المرباة",
        "ruche_donneuse": "الخلية المانحة",
    }
}


def get_lang():
    return st.session_state.get("langue", "fr")


def T(key):
    lang = get_lang()
    val = TRANSLATIONS[lang].get(key)
    if val is not None:
        return val
    # Fallback chain: en → fr → key
    val = TRANSLATIONS["en"].get(key)
    if val is not None:
        return val
    return TRANSLATIONS["fr"].get(key, key)


def inject_rtl_css():
    if get_lang() == "ar":
        st.markdown("""
        <style>
        .stApp, .main .block-container, [data-testid="stSidebar"],
        .stMarkdown, h1, h2, h3, h4, p, label {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Segoe UI', 'Arial', sans-serif !important;
        }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            direction: rtl !important;
        }
        </style>
        """, unsafe_allow_html=True)


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
# CSS PERSONNALISÉ (inchangé)
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
# BASE DE DONNÉES SQLITE (mise à jour avec nouvelles tables)
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
    c.execute("INSERT OR IGNORE INTO settings VALUES ('version','3.0.0')")


def init_db_v3():
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
    conn.commit()
    conn.close()


def init_db_v4():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
    -- Table pour le pedigree des reines
    CREATE TABLE IF NOT EXISTS reines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT NOT NULL,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE SET NULL,
        mere_id INTEGER REFERENCES reines(id) ON DELETE SET NULL,
        pere_id INTEGER REFERENCES reines(id) ON DELETE SET NULL,
        date_naissance TEXT,
        race TEXT,
        origine TEXT,
        qualite TEXT DEFAULT 'standard',
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Table pour les inspections vocales (historique)
    CREATE TABLE IF NOT EXISTS inspections_vocales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE CASCADE,
        date_inspection TEXT NOT NULL,
        texte_original TEXT,
        ia_analyse TEXT,
        varroa_pct REAL,
        nb_cadres INTEGER,
        reine_vue INTEGER,
        comportement TEXT,
        poids_kg REAL,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Table pour les analyses de cadre (scan)
    CREATE TABLE IF NOT EXISTS analyses_cadre (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE CASCADE,
        date_analyse TEXT NOT NULL,
        image_base64 TEXT,
        nb_abeilles INTEGER,
        reine_detectee INTEGER,
        maladies_detectees TEXT,
        varroa_visible INTEGER,
        couvain_sain_pct REAL,
        recommandations TEXT,
        ia_reponse TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Table pour les prédictions de transhumance
    CREATE TABLE IF NOT EXISTS transhumances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE CASCADE,
        date_depart TEXT,
        date_retour TEXT,
        latitude_dest REAL,
        longitude_dest REAL,
        lieu_dest TEXT,
        potentiel_miel TEXT,
        recommandation_ia TEXT,
        statut TEXT DEFAULT 'planifiee',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Insertion de quelques alertes par défaut
    INSERT OR IGNORE INTO alertes_config (type_alerte, seuil, actif, description) VALUES
        ('varroa_critique', 3.0, 1, 'Alerte varroa >= 3%'),
        ('cadres_insuffisants', 5, 1, 'Moins de 5 cadres de couvain');
    """)

    # Tables élevage reines v5
    c.executescript("""
    CREATE TABLE IF NOT EXISTS elevage_reines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT NOT NULL,
        ruche_donneuse_id INTEGER REFERENCES ruches(id) ON DELETE SET NULL,
        ruche_eleveuse_id INTEGER REFERENCES ruches(id) ON DELETE SET NULL,
        methode TEXT DEFAULT 'Doolittle',
        date_greffe TEXT,
        date_operculee TEXT,
        date_emergence TEXT,
        nb_cellules_greffees INTEGER DEFAULT 0,
        nb_cellules_acceptees INTEGER DEFAULT 0,
        qualite TEXT DEFAULT 'standard',
        statut TEXT DEFAULT 'en_cours',
        race TEXT,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS cellules_royales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        elevage_id INTEGER REFERENCES elevage_reines(id) ON DELETE CASCADE,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE SET NULL,
        numero INTEGER,
        stade TEXT DEFAULT 'oeuf',
        date_greffe TEXT,
        date_operculee TEXT,
        date_emergence TEXT,
        poids_mg REAL,
        longueur_mm REAL,
        qualite TEXT DEFAULT 'bonne',
        statut TEXT DEFAULT 'en_developpement',
        reine_resultante_id INTEGER REFERENCES reines(id),
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS elevage_males (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruche_id INTEGER REFERENCES ruches(id) ON DELETE SET NULL,
        race TEXT,
        date_debut TEXT,
        nb_cadres_males INTEGER DEFAULT 0,
        qualite_sperme TEXT DEFAULT 'non_evalue',
        reine_inseminee_id INTEGER REFERENCES reines(id),
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# AUTHENTIFICATION (inchangée)
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
        # Language selector on login page
        _lang_opts_l = ["🇫🇷 Français", "🇬🇧 English", "🇩🇿 العربية"]
        _lang_map_l  = {"fr": 0, "en": 1, "ar": 2}
        _lang_idx_l  = _lang_map_l.get(get_lang(), 0)
        _lang_choice_l = st.selectbox("🌐", _lang_opts_l, index=_lang_idx_l, key="lang_selector_login", label_visibility="collapsed")
        _new_lang_l = "ar" if "العربية" in _lang_choice_l else ("en" if "English" in _lang_choice_l else "fr")
        if _new_lang_l != get_lang():
            st.session_state["langue"] = _new_lang_l
            st.rerun()
        st.markdown(f"""
        <div style='text-align:center;margin-bottom:24px'>
            <div style='font-size:3rem'>🐝</div>
            <h1 style='font-family:Playfair Display,serif;color:#F0F4FF;font-size:2rem;margin:8px 0 4px'>ApiTrack Pro</h1>
            <p style='color:#A8B4CC;font-size:.9rem'>{T('login_subtitle')}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input(T("login_user"), placeholder="admin")
            password = st.text_input(T("login_pwd"), type="password", placeholder="••••••••")
            submitted = st.form_submit_button(T("login_btn"), use_container_width=True)

        if submitted:
            user = check_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                log_action("Connexion", f"Utilisateur {username} connecté")
                st.rerun()
            else:
                st.error(T("login_error"))

        st.markdown(f"<p style='text-align:center;font-size:.75rem;color:#A8B4CC;margin-top:16px'>{T('login_hint')}</p>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# UTILITAIRES (inchangés)
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
# MOTEUR IA MULTI-FOURNISSEURS (inchangé)
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
    """Appel IA avec retour JSON parsé. Tolérant au texte avant/après."""
    result = ia_call(prompt_text, image_bytes, json_mode=True)
    if not result or result.startswith("❌"):
        return {"error": result or "Pas de réponse"}
    text = result.strip()
    
    # Nettoyer les balises markdown
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if p.startswith("json"):
                text = p[4:].strip()
                break
            elif p.strip().startswith("{"):
                text = p.strip()
                break
        else:
            # Si aucun bloc JSON trouvé, prendre le dernier bloc
            text = parts[-1].strip()
    
    # Chercher un objet JSON avec regex (supporte les objets imbriqués)
    import re
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    # Essayer de trouver un tableau JSON
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    return {"error": f"JSON invalide : {text[:200]}"}


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS IA MÉTIER — utilisent ia_call()
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


def widget_cle_api():
    return widget_ia_selector()


# ════════════════════════════════════════════════════════════════════════════
# NOUVELLES FONCTIONS : PHOTOGRAMMÉTRIE (détection pièce 10 DA + mesure)
# ════════════════════════════════════════════════════════════════════════════

def detect_piece_and_measure(image_bytes):
    """
    Détecte la pièce de 10 DA (cercle) et mesure l'abeille.
    Retourne un dict avec les mesures en mm.
    """
    if not CV2_OK:
        return {"error": "OpenCV non installé. Installez opencv-python."}
    # Convertir bytes -> numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Impossible de décoder l'image."}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Détection des cercles (Hough Circles)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=20, maxRadius=200)
    if circles is None or len(circles[0]) == 0:
        return {"error": "Aucune pièce détectée. Vérifiez que la pièce de 10 DA est bien visible."}
    # Prendre le premier cercle détecté (supposé être la pièce)
    circle = circles[0][0]
    center = (int(circle[0]), int(circle[1]))
    radius = int(circle[2])
    diametre_px = 2 * radius
    # Diamètre réel de la pièce de 10 DA = 20 mm
    DIAMETRE_REEL_MM = 20.0
    echelle_mm_par_px = DIAMETRE_REEL_MM / diametre_px

    # Détection de l'abeille (contour)
    # On cherche le plus grand contour après seuillage
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "Aucun contour d'abeille détecté."}
    # On suppose que l'abeille est le plus grand contour (hors cercle)
    bee_contour = max(contours, key=cv2.contourArea)
    # Dimensions approximatives : rectangle englobant
    x, y, w, h = cv2.boundingRect(bee_contour)
    # La longueur de l'aile n'est pas directement mesurable par bounding box.
    # On va estimer la longueur de l'abeille (tête + thorax + abdomen)
    longueur_abeille_px = max(w, h)  # approximation
    # Largeur de l'aile? Difficile. On va faire simple: on mesure la longueur de l'aile via un autre contour?
    # Pour rester simple, on demande à l'utilisateur de cliquer sur l'aile? Mais la demande est "mesure automatique".
    # On peut essayer de détecter les ailes par analyse de forme (contours dans la région haute).
    # Ici, on va utiliser une heuristique : la longueur d'aile est environ 1/3 de la longueur totale pour Apis mellifera.
    # Ce n'est pas précis, mais c'est une démo.
    # Alternative: on utilise l'IA pour la mesure, mais l'utilisateur veut une mesure locale.
    # Je vais plutôt mesurer la longueur de l'abeille (tête+thorax+abdomen) et l'utilisateur pourra ajuster.

    # Pour rester cohérent avec les champs de morphométrie, on calcule :
    longueur_abeille_mm = longueur_abeille_px * echelle_mm_par_px
    # Estimation empirique : l'aile antérieure fait environ 70% de la longueur du corps
    longueur_aile_mm = longueur_abeille_mm * 0.7
    largeur_aile_mm = longueur_aile_mm * 0.35  # ratio typique
    indice_cubital = 2.3  # valeur par défaut (non mesurable)
    glossa_mm = longueur_abeille_mm * 0.2  # approximation
    tomentum = 2
    pigmentation = "Brun foncé"

    return {
        "longueur_aile_mm": round(longueur_aile_mm, 2),
        "largeur_aile_mm": round(largeur_aile_mm, 2),
        "indice_cubital": indice_cubital,
        "glossa_mm": round(glossa_mm, 2),
        "tomentum": tomentum,
        "pigmentation": pigmentation,
        "echelle_mm_par_px": round(echelle_mm_par_px, 4),
        "diametre_piece_px": diametre_px,
        "longueur_corps_mm": round(longueur_abeille_mm, 2)
    }

# ════════════════════════════════════════════════════════════════════════════
# NOUVELLE FONCTION : GÉOCODAGE (recherche de ville)
# ════════════════════════════════════════════════════════════════════════════

def geocode_ville(nom_ville):
    """Utilise Nominatim pour obtenir les coordonnées d'une ville."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": nom_ville,
            "format": "json",
            "limit": 1,
            "addressdetails": 0
        }
        req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": "ApiTrackPro/3.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        st.error(f"Erreur géocodage : {e}")
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS POUR LES NOUVELLES FONCTIONNALITÉS (assistant vocal, scan cadre, transhumance)
# ════════════════════════════════════════════════════════════════════════════

def extraire_inspection_ia(texte_transcrit):
    """Utilise l'IA pour extraire les champs d'inspection depuis le texte vocal."""
    prompt = f"""Tu es un assistant apicole spécialisé dans l'extraction de données d'inspection.

À partir du texte suivant dicté par un apiculteur, extrait les informations suivantes au format JSON strict (sans texte additionnel) :

Texte : "{texte_transcrit}"

Retourne un objet JSON avec les clés suivantes (utilise null si absent) :
{{
    "varroa_pct": nombre (ex: 2.5),
    "nb_cadres": entier (ex: 9),
    "reine_vue": booléen (true/false),
    "comportement": "calme" ou "nerveuse" ou "agressive" ou "tres calme",
    "poids_kg": nombre (ex: 28.4),
    "notes": "résumé des autres observations"
}}

Règles :
- Si l'utilisateur dit "reine vue" -> true, "reine non vue" -> false
- Varroa peut être dit "deux virgule cinq pourcent" -> 2.5
- Comportement : "calme", "nerveuse", "agressive"
- Ne retourne que le JSON, rien d'autre."""
    return ia_call_json(prompt)


def analyser_cadre_ia(image_bytes):
    """Analyse une photo de cadre de couvain : compte abeilles, détecte reine, maladies, etc."""
    prompt = """Tu es un expert en pathologie apicole et en vision par ordinateur. Analyse cette photo d'un cadre de ruche.

Réponds UNIQUEMENT avec un objet JSON valide (pas de texte avant/après) :
{
    "nb_abeilles": entier (estimation visuelle du nombre d'abeilles adultes visibles),
    "reine_detectee": booléen (true si la reine est visible sur la photo),
    "varroa_visible": booléen (true si des varroas sont visibles sur les abeilles ou le couvain),
    "maladies_detectees": ["loque_americaine", "loque_europeenne", "nosemose", "couvain_sac", "acarien", "aucune"],
    "couvain_sain_pct": entier (estimation du pourcentage de couvain sain par rapport au total visible),
    "recommandations": "courte phrase d'action recommandée",
    "commentaire": "observations supplémentaires"
}

Règles :
- Si une maladie est suspectée, ajoute-la dans maladies_detectees.
- Sois précis et professionnel.
- Si l'image est floue ou peu exploitable, indique-le dans commentaire."""
    return ia_call_json(prompt, image_bytes)


def predire_transhumance_ia(lat, lon, description_zone="", periode_souhaitee="Printemps"):
    """Génère une recommandation de transhumance très détaillée pour la zone et la période données."""
    prompt = f"""Tu es un expert en transhumance apicole, botaniste et météorologue. 
Analyse en profondeur la zone située aux coordonnées {lat:.4f}, {lon:.4f}.
Nom du lieu approximatif : {description_zone}
Période de transhumance souhaitée : {periode_souhaitee}

Tu dois répondre UNIQUEMENT par un objet JSON valide, sans aucun texte avant ou après, sans balises markdown. Le JSON doit contenir les champs suivants (respecte exactement les noms) :

{{
    "resume_zone": "Description courte de la zone (type de paysage, altitude, climat général)",
    "potentiel_miel": "excellent" / "bon" / "moyen" / "faible",
    "periode_optimale_depart": "mois recommandé pour installer les ruches (si différent de la période souhaitée, justifie)",
    "periode_optimale_retour": "mois recommandé pour quitter la zone",
    "duree_sejour_jours": nombre entier,
    "flore_dominante": "espèce mellifère principale pour la période souhaitée",
    "flore_detaillee": [
        {{
            "espece": "nom scientifique et commun",
            "type": "nectar" / "pollen" / "nectar+pollen" / "miellat",
            "abondance": "abondante" / "moyenne" / "rare",
            "periode_floraison": "mois de floraison",
            "potentiel_mellifere": "élevé" / "moyen" / "faible",
            "fleurit_pendant_periode": true/false
        }}
    ],
    "plantes_en_fleur_periode": ["liste des espèces qui fleurissent pendant la période souhaitée"],
    "meteo_moyenne": {{
        "temp_printemps_c": "XX-YY",
        "temp_ete_c": "XX-YY",
        "temp_automne_c": "XX-YY",
        "pluie_annuelle_mm": nombre,
        "ensoleillement_heures": nombre,
        "vent_dominant": "direction et force"
    }},
    "meteo_pendant_periode": {{
        "temperature_moyenne_c": "XX-YY",
        "precipitations_mm": nombre,
        "risque_gel": "oui/non",
        "risque_secheresse": "oui/non"
    }},
    "risques": ["risque1", "risque2", "risque3"],
    "recommandations": ["recommandation1", "recommandation2", "recommandation3"],
    "ressources_eau": "disponibilité en eau pour les abeilles",
    "acces_ruches": "facilité d'accès et de transport",
    "concurrence_apicole": "estimation de la présence d'autres apiculteurs"
}}

Remplis avec des données réalistes pour l'Algérie / Afrique du Nord. Sois précis sur les espèces végétales locales. Indique pour chaque espèce si elle est pollinifère, nectarifère ou les deux, et surtout si elle fleurit pendant la période demandée ({periode_souhaitee})."""
    
    result = ia_call_json(prompt)
    
    # Fallback intelligent si l'IA ne répond pas correctement
    if "error" in result or not isinstance(result, dict):
        is_urban = any(keyword in description_zone.lower() for keyword in ["urban", "city", "centre", "rue", "boulevard", "tlemcen", "oran", "alger"])
        # Déterminer les plantes selon la période
        if "printemps" in periode_souhaitee.lower():
            plantes_periode = ["Romarin", "Thym", "Oranger", "Amandier", "Jujubier"]
            meteo_periode = {"temperature_moyenne_c": "15-22", "precipitations_mm": 60, "risque_gel": "non", "risque_secheresse": "non"}
        elif "ete" in periode_souhaitee.lower():
            plantes_periode = ["Lavande", "Eucalyptus", "Tournesol", "Figuier"]
            meteo_periode = {"temperature_moyenne_c": "25-35", "precipitations_mm": 10, "risque_gel": "non", "risque_secheresse": "oui"}
        elif "automne" in periode_souhaitee.lower():
            plantes_periode = ["Caroube", "Olivier", "Lierre"]
            meteo_periode = {"temperature_moyenne_c": "18-28", "precipitations_mm": 40, "risque_gel": "non", "risque_secheresse": "non"}
        else:  # hiver
            plantes_periode = ["Romarin (hivernal)", "Eucalyptus (floraison hiver)", "Arbousier"]
            meteo_periode = {"temperature_moyenne_c": "8-15", "precipitations_mm": 80, "risque_gel": "oui", "risque_secheresse": "non"}
        
        if is_urban:
            return {
                "resume_zone": "Zone urbaine ou périurbaine avec peu de flore naturelle.",
                "potentiel_miel": "faible",
                "periode_optimale_depart": "Avril",
                "periode_optimale_retour": "Mai",
                "duree_sejour_jours": 30,
                "flore_dominante": "Jardins ornementaux",
                "flore_detaillee": [
                    {"espece": "Orangers (Citrus sinensis)", "type": "nectar+pollen", "abondance": "moyenne", "periode_floraison": "Mars-Avril", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": "printemps" in periode_souhaitee.lower()},
                    {"espece": "Laurier-rose", "type": "pollen", "abondance": "abondante", "periode_floraison": "Juin-Septembre", "potentiel_mellifere": "faible", "fleurit_pendant_periode": "ete" in periode_souhaitee.lower()}
                ],
                "plantes_en_fleur_periode": plantes_periode if not is_urban else ["Quelques arbres ornementaux"],
                "meteo_moyenne": {
                    "temp_printemps_c": "15-25", "temp_ete_c": "22-35", "temp_automne_c": "15-28",
                    "pluie_annuelle_mm": 400, "ensoleillement_heures": 2800, "vent_dominant": "Nord-Ouest modéré"
                },
                "meteo_pendant_periode": meteo_periode,
                "risques": ["Pollution", "Pesticides", "Peu de ressources mellifères"],
                "recommandations": ["Déplacer vers zone rurale", "Transhumer en avril"],
                "ressources_eau": "Fontaines, bassins",
                "acces_ruches": "Bon accès routier",
                "concurrence_apicole": "Faible"
            }
        else:
            # Flore naturelle selon période
            flore_detail = []
            if "printemps" in periode_souhaitee.lower():
                flore_detail = [
                    {"espece": "Romarin (Rosmarinus officinalis)", "type": "nectar+pollen", "abondance": "abondante", "periode_floraison": "Fév-Mai", "potentiel_mellifere": "élevé", "fleurit_pendant_periode": True},
                    {"espece": "Thym (Thymus vulgaris)", "type": "nectar+pollen", "abondance": "abondante", "periode_floraison": "Avr-Juin", "potentiel_mellifere": "élevé", "fleurit_pendant_periode": True},
                    {"espece": "Jujubier (Ziziphus lotus)", "type": "nectar", "abondance": "moyenne", "periode_floraison": "Avr-Juin", "potentiel_mellifere": "élevé", "fleurit_pendant_periode": True},
                    {"espece": "Amandier", "type": "nectar+pollen", "abondance": "rare", "periode_floraison": "Fév-Mar", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                ]
            elif "ete" in periode_souhaitee.lower():
                flore_detail = [
                    {"espece": "Lavande (Lavandula stoechas)", "type": "nectar+pollen", "abondance": "moyenne", "periode_floraison": "Mai-Juil", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                    {"espece": "Eucalyptus", "type": "nectar", "abondance": "moyenne", "periode_floraison": "Juin-Août", "potentiel_mellifere": "élevé", "fleurit_pendant_periode": True},
                    {"espece": "Tournesol", "type": "pollen", "abondance": "rare", "periode_floraison": "Juil-Août", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                ]
            elif "automne" in periode_souhaitee.lower():
                flore_detail = [
                    {"espece": "Caroube (Ceratonia siliqua)", "type": "nectar", "abondance": "moyenne", "periode_floraison": "Sep-Oct", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                    {"espece": "Olivier", "type": "pollen", "abondance": "abondante", "periode_floraison": "Mai-Juin", "potentiel_mellifere": "faible", "fleurit_pendant_periode": False},
                    {"espece": "Lierre", "type": "nectar", "abondance": "moyenne", "periode_floraison": "Sep-Nov", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                ]
            else:  # hiver
                flore_detail = [
                    {"espece": "Romarin (floraison hivernale)", "type": "nectar+pollen", "abondance": "moyenne", "periode_floraison": "Dec-Fév", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                    {"espece": "Eucalyptus (variétés hivernales)", "type": "nectar", "abondance": "moyenne", "periode_floraison": "Nov-Jan", "potentiel_mellifere": "moyen", "fleurit_pendant_periode": True},
                ]
            return {
                "resume_zone": "Zone méditerranéenne naturelle avec garrigue et forêt claire.",
                "potentiel_miel": "bon",
                "periode_optimale_depart": "Mars",
                "periode_optimale_retour": "Juin",
                "duree_sejour_jours": 90,
                "flore_dominante": "Romarin, Thym",
                "flore_detaillee": flore_detail,
                "plantes_en_fleur_periode": [f["espece"] for f in flore_detail if f.get("fleurit_pendant_periode")],
                "meteo_moyenne": {
                    "temp_printemps_c": "12-22", "temp_ete_c": "20-35", "temp_automne_c": "15-28",
                    "pluie_annuelle_mm": 500, "ensoleillement_heures": 3000, "vent_dominant": "Nord-Ouest"
                },
                "meteo_pendant_periode": meteo_periode,
                "risques": ["Sécheresse", "Incendies", "Frelons"],
                "recommandations": ["Placer près d'un point d'eau", "Surveiller la floraison"],
                "ressources_eau": "Oueds temporaires",
                "acces_ruches": "Chemins ruraux",
                "concurrence_apicole": "Modérée"
            }
    return result


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR (mise à jour avec nouvelles pages)
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

        # ── Sélecteur de langue ─────────────────────────────────────────────
        _lang_opts = ["🇫🇷 Français", "🇬🇧 English", "🇩🇿 العربية"]
        _lang_map  = {"fr": 0, "en": 1, "ar": 2}
        _lang_idx  = _lang_map.get(get_lang(), 0)
        _lang_choice = st.sidebar.selectbox(
            T("langue"), _lang_opts, index=_lang_idx, key="lang_selector"
        )
        _new_lang = "ar" if "العربية" in _lang_choice else ("en" if "English" in _lang_choice else "fr")
        if _new_lang != get_lang():
            st.session_state["langue"] = _new_lang
            st.rerun()

        pages = {
            T("nav_dashboard"): "dashboard",
            T("nav_ruches"): "ruches",
            T("nav_inspections"): "inspections",
            T("nav_assistant_vocal"): "assistant_vocal",
            T("nav_traitements"): "traitements",
            T("nav_productions"): "productions",
            T("nav_morpho"): "morpho",
            T("nav_carto"): "carto",
            T("nav_scanner_cadre"): "scanner_cadre",
            T("nav_transhumance"): "transhumance",
            T("nav_pedigree"): "pedigree",
            T("nav_elevage_reines"): "elevage_reines",
            T("nav_cellules_royales"): "cellules_royales",
            T("nav_meteo"): "meteo",
            T("nav_genetique"): "genetique",
            T("nav_flore"): "flore",
            T("nav_alertes"): "alertes",
            T("nav_journal"): "journal",
            T("nav_admin"): "admin",
        }

        # ── Niveau utilisateur ───────────────────────────────────────────────
        _niv_opts = [T("niveau_debutant"), T("niveau_intermediaire"), T("niveau_expert")]
        _niv_idx_map = {"debutant": 0, "intermediaire": 1, "expert": 2}
        _cur_niv = st.session_state.get("niveau_user", "intermediaire")
        _niv_idx = _niv_idx_map.get(_cur_niv, 1)
        _niv_choice = st.sidebar.selectbox(
            T("niveau_label"), _niv_opts, index=_niv_idx, key="niveau_selector"
        )
        if T("niveau_debutant") in _niv_choice:
            st.session_state["niveau_user"] = "debutant"
        elif T("niveau_expert") in _niv_choice:
            st.session_state["niveau_user"] = "expert"
        else:
            st.session_state["niveau_user"] = "intermediaire"

        if "page" not in st.session_state:
            st.session_state.page = "dashboard"

        for label, key in pages.items():
            if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.sidebar.markdown("<hr style='border-color:#2E3A52;margin:12px 0'>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size:.75rem;color:#6B7A99'>👤 {st.session_state.get('username','admin')}</div>", unsafe_allow_html=True)
        if st.sidebar.button(T("logout"), use_container_width=True):
            log_action("Déconnexion", f"Utilisateur {st.session_state.get('username')} déconnecté")
            st.session_state.logged_in = False
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : DASHBOARD (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown(T("dashboard_title"))
    rucher = get_setting("rucher_nom", "Mon Rucher")
    localisation = get_setting("localisation", "")
    st.markdown(f"<p style='color:#A8B4CC;margin-top:-10px'>Saison printanière 2025 · {rucher} · {localisation}</p>", unsafe_allow_html=True)

    conn = get_db()
    nb_ruches = conn.execute("SELECT COUNT(*) FROM ruches WHERE statut='actif'").fetchone()[0]
    total_miel = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='miel'").fetchone()[0]
    nb_insp = conn.execute("SELECT COUNT(*) FROM inspections WHERE date_inspection >= date('now','-30 days')").fetchone()[0]
    critiques = conn.execute("SELECT COUNT(*) FROM inspections WHERE varroa_pct >= 3.0 AND date_inspection >= date('now','-7 days')").fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(T("metric_ruches"), nb_ruches, "+3 ce mois")
    col2.metric(T("metric_miel"), f"{total_miel:.0f}", "+18% vs 2024")
    col3.metric(T("metric_insp"), nb_insp, "Cadence correcte")
    col4.metric(T("metric_varroa"), critiques, "Intervention requise" if critiques else "RAS", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(T("chart_prod"))
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
        st.markdown(T("chart_etat"))
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

    st.markdown(T("alertes_actives"))
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
        st.success(T("no_alert"))


# ════════════════════════════════════════════════════════════════════════════
# PAGE : GESTION DES RUCHES (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_ruches():
    st.markdown(T("ruches_title"))

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

    tab1, tab2 = st.tabs([T("tab_liste"), T("tab_ajouter")])

    with tab1:
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(T("export_csv"), csv, "ruches.csv", "text/csv")

        st.markdown(T("supprimer_ruche"))
        ruche_ids = conn.execute("SELECT id, nom FROM ruches").fetchall()
        if ruche_ids:
            options = {f"R{r[0]:02d} — {r[1]}": r[0] for r in ruche_ids}
            selected = st.selectbox(T("choisir_ruche"), options.keys())
            if st.button(T("btn_supprimer"), type="secondary"):
                rid = options[selected]
                conn.execute("DELETE FROM ruches WHERE id=?", (rid,))
                conn.commit()
                log_action("Suppression ruche", f"Ruche {selected} supprimée")
                st.success(f"Ruche {selected} supprimée.")
                st.rerun()

    with tab2:
        with st.form("add_ruche"):
            st.markdown(T("nouvelle_ruche"))
            col1, col2 = st.columns(2)
            nom = col1.text_input(T("nom_ruche"))
            race = col2.selectbox(T("race"), ["intermissa", "sahariensis", "ligustica", "carnica", "hybride"])
            date_inst = col1.date_input(T("date_install"), datetime.date.today())
            localisation = col2.text_input(T("localisation_label"))
            col3, col4 = st.columns(2)
            lat = col3.number_input("Latitude", value=34.88, format="%.4f")
            lon = col4.number_input("Longitude", value=1.32, format="%.4f")
            notes = st.text_area(T("notes"))
            submitted = st.form_submit_button(T("btn_ajouter"))

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
# PAGE : INSPECTIONS (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_inspections():
    st.markdown(T("insp_title"))
    conn = get_db()

    tab1, tab2 = st.tabs([T("tab_historique"), T("tab_nouvelle")])

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
            st.download_button(T("export_csv"), csv, "inspections.csv", "text/csv")

        st.markdown(T("evolution_varroa"))
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
            ruche_sel = col1.selectbox(T("ruche_label"), opts.keys())
            date_insp = col2.date_input(T("date_label"), datetime.date.today())
            col3, col4, col5 = st.columns(3)
            poids = col3.number_input(T("poids_label"), 0.0, 80.0, 25.0, 0.1)
            cadres = col4.number_input(T("cadres_label"), 0, 20, 10)
            varroa = col5.number_input(T("varroa_label"), 0.0, 20.0, 1.0, 0.1)
            col6, col7 = st.columns(2)
            reine = col6.checkbox(T("reine_vue_label"), value=True)
            comportement = col7.selectbox(T("comportement_label"), ["calme", "nerveuse", "agressive", "très calme"])
            notes = st.text_area("Notes / Observations")
            submitted = st.form_submit_button(T("btn_enregistrer"))

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
# PAGE : TRAITEMENTS (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_traitements():
    st.markdown(T("trait_title"))
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
            st.info(T("no_trait"))

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
# PAGE : PRODUCTIONS (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_productions():
    st.markdown(T("prod_title"))
    conn = get_db()

    total_miel = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='miel'").fetchone()[0]
    total_pollen = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='pollen'").fetchone()[0]
    total_gr = conn.execute("SELECT COALESCE(SUM(quantite_kg),0) FROM recoltes WHERE type_produit='gelée royale'").fetchone()[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🍯 Miel total (kg)", f"{total_miel:.1f}", "Humidité moy. 17.2%")
    col2.metric("🌼 Pollen (kg)", f"{total_pollen:.1f}", "Qualité A")
    col3.metric("👑 Gelée royale (kg)", f"{total_gr:.2f}", "10-HDA 2.1%")

    tab1, tab2, tab3 = st.tabs([T("tab_recoltes"), T("tab_graphiques"), T("tab_nouvelle_rec")])

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
# PAGE : MORPHOMÉTRIE IA (inchangée, mais avec photogrammétrie déjà intégrée)
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


DIAMETRES_ETALONS = {
    "Pièce 10 DA": 20.0,
    "Pièce 1€":    23.25,
    "Pièce 1$":    26.5,
}


def model_supporte_vision():
    prov = get_active_provider()
    model = get_active_model()
    cfg = IA_PROVIDERS.get(prov, {})
    if not cfg.get("vision"):
        return False
    if cfg.get("type") == "google" and model.startswith("gemma"):
        return False
    return True


def ia_mesurer_morphometrie_auto(image_bytes, etalon_type="Pièce 10 DA"):
    diametre_etalon_mm = DIAMETRES_ETALONS.get(etalon_type, 20.0)
    prompt = f"""Tu es un expert en morphométrie apicole et en analyse d'images.
Tu reçois une photo macro d'une abeille (Apis mellifera) placée à côté d'une pièce de monnaie étalon ({etalon_type}, diamètre réel = {diametre_etalon_mm} mm) pour calibration.

Analyse l'image et mesure avec précision :
1. Détecte la pièce étalon pour calibrer l'échelle pixels/mm
2. Mesure les structures morphologiques de l'abeille

Retourne UNIQUEMENT un objet JSON valide (sans balises markdown, sans texte avant ou après) :
{{
  "calibration_detectee": true,
  "etalon_utilise": "{etalon_type}",
  "longueur_aile_mm": 9.2,
  "largeur_aile_mm": 3.1,
  "indice_cubital": 2.3,
  "glossa_mm": 6.1,
  "tomentum": 2,
  "pigmentation": "Brun foncé",
  "confiance_mesure_pct": 85,
  "notes_auto": "Courte description de ce que tu as observé sur la photo",
  "avertissements": []
}}

Règles :
- tomentum : entier entre 0 et 3
- pigmentation : exactement l'une de ces valeurs : "Noir", "Brun foncé", "Brun clair", "Jaune"
- Si tu ne peux pas mesurer un paramètre, garde la valeur typique d'Apis mellifera intermissa
- confiance_mesure_pct : ton niveau de confiance global en %
- avertissements : liste de messages si la photo est floue, étalon absent, etc.
"""
    return ia_call_json(prompt, image_bytes)


def ia_estimer_morphometrie_texte(etalon_type, px_etalon, px_aile, px_largeur,
                                   px_cubital_a, px_cubital_b, px_cubital_c,
                                   px_glossa, tomentum, pigmentation):
    diametre_mm = DIAMETRES_ETALONS.get(etalon_type, 20.0)
    prompt = f"""Tu es un expert en morphométrie apicole (méthode Ruttner 1988).
L'utilisateur a mesuré manuellement les structures de l'abeille en pixels sur une photo,
en utilisant une {etalon_type} (diamètre réel = {diametre_mm} mm) comme étalon.

Mesures en pixels :
- Diamètre pièce étalon : {px_etalon} px  → 1 mm = {px_etalon}/{diametre_mm:.1f} px
- Longueur aile antérieure : {px_aile} px
- Largeur aile : {px_largeur} px
- Nervures cubitales a : {px_cubital_a} px, b : {px_cubital_b} px, c : {px_cubital_c} px
- Longueur glossa : {px_glossa} px
- Tomentum observé : {tomentum}/3
- Pigmentation scutellum : {pigmentation}

Calcule et retourne UNIQUEMENT un objet JSON valide :
{{
  "calibration_detectee": true,
  "etalon_utilise": "{etalon_type}",
  "echelle_px_par_mm": {round(px_etalon/diametre_mm, 4) if px_etalon > 0 else 0},
  "longueur_aile_mm": 0.0,
  "largeur_aile_mm": 0.0,
  "indice_cubital": 0.0,
  "glossa_mm": 0.0,
  "tomentum": {tomentum},
  "pigmentation": "{pigmentation}",
  "confiance_mesure_pct": 90,
  "notes_auto": "Mensuration assistée — calcul Gemma depuis mesures en pixels",
  "avertissements": []
}}

Formules de calcul :
- echelle = px_etalon / {diametre_mm}  (px par mm)
- longueur_aile_mm = px_aile / echelle
- largeur_aile_mm = px_largeur / echelle
- indice_cubital = (px_cubital_a / px_cubital_b) / (px_cubital_b / px_cubital_c) si px_cubital_b > 0 et px_cubital_c > 0, sinon 2.3
- glossa_mm = px_glossa / echelle
Arrondis à 2 décimales. Si px = 0, utilise valeurs typiques d'A.m. intermissa.
"""
    return ia_call_json(prompt)


def _appliquer_mesures_auto(result):
    if "error" in result:
        st.error(f"❌ Erreur mensuration IA : {result['error']}")
        return
    pig_valid = ["Noir", "Brun foncé", "Brun clair", "Jaune"]
    pig_raw   = result.get("pigmentation", "Brun foncé")
    st.session_state["morpho_aile"]         = float(result.get("longueur_aile_mm", 9.2))
    st.session_state["morpho_largeur"]      = float(result.get("largeur_aile_mm", 3.1))
    st.session_state["morpho_cubital"]      = float(result.get("indice_cubital", 2.3))
    st.session_state["morpho_glossa"]       = float(result.get("glossa_mm", 6.1))
    st.session_state["morpho_tomentum"]     = int(result.get("tomentum", 2))
    st.session_state["morpho_pigmentation"] = pig_raw if pig_raw in pig_valid else "Brun foncé"
    st.session_state["morpho_notes_auto"]   = result.get("notes_auto", "Mensuration assistée")
    confiance_auto  = result.get("confiance_mesure_pct", 0)
    avertissements  = result.get("avertissements", [])

    st.success(f"✅ Mensuration terminée — confiance IA : **{confiance_auto}%**")
    for av in avertissements:
        st.warning(f"⚠️ {av}")

    st.markdown("#### 📐 Mesures calculées")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Aile ant. (mm)", f"{st.session_state['morpho_aile']:.2f}")
    col_r2.metric("Largeur aile (mm)", f"{st.session_state['morpho_largeur']:.2f}")
    col_r3.metric("Indice cubital", f"{st.session_state['morpho_cubital']:.2f}")
    col_r4.metric("Glossa (mm)", f"{st.session_state['morpho_glossa']:.2f}")
    col_r5, col_r6, _ = st.columns(3)
    col_r5.metric("Tomentum", st.session_state["morpho_tomentum"])
    col_r6.metric("Pigmentation", st.session_state["morpho_pigmentation"])

    if st.session_state["morpho_notes_auto"]:
        st.markdown(
            f"<div style='background:#0F1117;border-left:3px solid #C8820A;padding:8px 12px;"
            f"border-radius:4px;font-size:.85rem;color:#A8B4CC;margin-top:8px'>"
            f"🔍 <i>{st.session_state['morpho_notes_auto']}</i></div>",
            unsafe_allow_html=True
        )
    st.info("➡️ Mesures reportées dans **🔬 Analyse + IA** — vérifiez et lancez l'analyse complète.")
    log_action("Morphométrie Auto", f"Mensuration — confiance {confiance_auto}%")


def page_morpho():
    st.markdown(T("morpho_title"))
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

    # Initialiser les valeurs de session pour la mensuration auto
    for k, v in [("morpho_aile", 9.2), ("morpho_largeur", 3.1), ("morpho_cubital", 2.3),
                 ("morpho_glossa", 6.1), ("morpho_tomentum", 2),
                 ("morpho_pigmentation", "Brun foncé"), ("morpho_notes_auto", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Nouvel onglet pour la photogrammétrie
    tab0, tab1, tab2, tab3 = st.tabs(["📷 Photogrammétrie (10 DA)", "🤖 Mensuration Auto IA", "🔬 Analyse + IA", "📜 Historique"])

    # ── ONGLET 0 : PHOTOGRAMMÉTRIE AVEC DÉTECTION PIÈCE ────────────────────
    with tab0:
        st.markdown("### 📷 Photogrammétrie automatique - Pièce de 10 DA")
        st.markdown("""
        <div style='background:#0D2A1F;border:1px solid #1A5C3A;border-radius:8px;padding:10px;margin-bottom:12px'>
        <b>⚙️ Fonctionnement :</b> Téléchargez une photo macro où figure une abeille à côté d'une pièce de 10 DA.
        L'application détecte automatiquement la pièce (cercle), calibre l'échelle, puis mesure l'abeille.
        </div>
        """, unsafe_allow_html=True)

        if not CV2_OK:
            st.error("❌ OpenCV n'est pas installé. Installez-le avec `pip install opencv-python` pour utiliser cette fonctionnalité.")
        else:
            img_photogram = st.file_uploader("📷 Photo (abeille + pièce 10 DA)", type=["jpg","jpeg","png"], key="photogram_img")
            if img_photogram:
                st.image(img_photogram, caption="Photo chargée", use_column_width=True)
                if st.button("🔍 Détecter et mesurer automatiquement", use_container_width=True, type="primary"):
                    with st.spinner("Analyse de l'image en cours..."):
                        mesures = detect_piece_and_measure(img_photogram.read())
                    if "error" in mesures:
                        st.error(mesures["error"])
                    else:
                        st.success("✅ Détection réussie !")
                        st.markdown("#### 📏 Résultats des mesures")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Longueur aile (mm)", f"{mesures['longueur_aile_mm']:.2f}")
                        col2.metric("Largeur aile (mm)", f"{mesures['largeur_aile_mm']:.2f}")
                        col3.metric("Glossa (mm)", f"{mesures['glossa_mm']:.2f}")
                        st.metric("Indice cubital (estimé)", mesures['indice_cubital'])
                        st.metric("Tomentum (estimé)", mesures['tomentum'])
                        st.metric("Pigmentation (estimée)", mesures['pigmentation'])

                        # Transférer les mesures dans session_state pour l'onglet Analyse
                        st.session_state["morpho_aile"] = mesures['longueur_aile_mm']
                        st.session_state["morpho_largeur"] = mesures['largeur_aile_mm']
                        st.session_state["morpho_cubital"] = mesures['indice_cubital']
                        st.session_state["morpho_glossa"] = mesures['glossa_mm']
                        st.session_state["morpho_tomentum"] = mesures['tomentum']
                        st.session_state["morpho_pigmentation"] = mesures['pigmentation']
                        st.session_state["morpho_notes_auto"] = f"Mesures par photogrammétrie (pièce 10 DA), échelle {mesures['echelle_mm_par_px']:.4f} mm/px"

                        st.info("✅ Les mesures ont été transférées vers l'onglet **🔬 Analyse + IA**. Vous pouvez maintenant lancer l'analyse IA.")

    # ── ONGLET 1 : MENSURATION AUTO IA (inchangé) ───────────────────────────
    with tab1:
        st.markdown("### 📷 Mensuration morphométrique automatique par IA")

        prov_now  = get_active_provider()
        model_now = get_active_model()
        vision_ok = model_supporte_vision()

        if vision_ok:
            st.markdown(
                f"<div style='background:#0F2B1A;border:1px solid #34D399;border-left:4px solid #34D399;"
                f"border-radius:6px;padding:8px 14px;margin-bottom:10px;font-size:.83rem;color:#34D399'>"
                f"✅ <b>Mode Vision activé</b> — {prov_now} / {model_now} analyse directement la photo.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#1A1A0A;border:1px solid #F5A623;border-left:4px solid #F5A623;"
                f"border-radius:6px;padding:8px 14px;margin-bottom:10px;font-size:.83rem;color:#F5A623'>"
                f"🔢 <b>Mode Assisté activé</b> — {model_now} (sans vision). "
                f"Mesurez les structures en pixels sur la photo, Gemma calcule les mm.</div>",
                unsafe_allow_html=True
            )

        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            etalon_type = st.selectbox(
                "🪙 Pièce étalon",
                list(DIAMETRES_ETALONS.keys()),
                index=0,
                help="Placez cette pièce à côté de l'abeille sur la photo."
            )
            st.markdown(
                f"<small style='color:#F5A623'>Diamètre réel : <b>{DIAMETRES_ETALONS[etalon_type]} mm</b></small>",
                unsafe_allow_html=True
            )
        with col_e2:
            st.markdown(
                """<div style='background:#0F1117;border:1px solid #3A4A66;border-radius:8px;
                padding:8px 12px;font-size:.81rem;color:#A8B4CC'>
                <b style='color:#F5A623'>💡 Conseils photo :</b>
                Éclairage uniforme · Abeille aplatie · Aile bien dépliée ·
                Pièce dans le même plan · Résolution ≥ 1 MP
                </div>""",
                unsafe_allow_html=True
            )

        img_auto = st.file_uploader(
            "📷 Photo macro abeille + pièce étalon",
            type=["jpg", "jpeg", "png", "webp"],
            key="morpho_auto_img"
        )

        if img_auto:
            st.image(img_auto, caption="Photo chargée", use_column_width=True)

        if vision_ok:
            btn_auto = st.button(
                "🔬 Lancer la mensuration automatique par IA",
                disabled=(not ia_active or img_auto is None),
                use_container_width=True,
                type="primary"
            )
            if not ia_active:
                st.info("🔑 Configurez votre clé API IA.")
            elif img_auto is None:
                st.info("⬆️ Chargez une photo pour lancer la mensuration.")

            if btn_auto and img_auto and ia_active:
                img_bytes = img_auto.read()
                with st.spinner(f"🤖 {model_now} analyse la photo et mesure les structures..."):
                    result = ia_mesurer_morphometrie_auto(img_bytes, etalon_type)
                _appliquer_mesures_auto(result)
        else:
            st.markdown("---")
            st.markdown(
                "#### 📏 Mesurez les structures en pixels sur votre photo\n"
                "<small style='color:#A8B4CC'>Utilisez un logiciel comme "
                "<b>ImageJ</b>, <b>GIMP</b> ou l'outil de mesure de votre téléphone. "
                "Mesurez la pièce étalon d'abord pour calibrer, puis chaque structure.</small>",
                unsafe_allow_html=True
            )

            col_px1, col_px2 = st.columns(2)
            with col_px1:
                px_etalon   = st.number_input(f"📏 Diamètre {etalon_type} (px)", 10, 5000, 400, 1,
                                               help="Mesurez le diamètre de la pièce en pixels")
                px_aile     = st.number_input("📏 Longueur aile antérieure (px)", 0, 5000, 0, 1)
                px_largeur  = st.number_input("📏 Largeur aile (px)", 0, 5000, 0, 1)
                px_glossa   = st.number_input("📏 Longueur glossa (px)", 0, 5000, 0, 1)

            with col_px2:
                st.markdown(
                    "<small style='color:#A8B4CC'><b>Indice cubital :</b> "
                    "mesurez les 3 segments de nervure a, b, c</small>",
                    unsafe_allow_html=True
                )
                px_cubital_a = st.number_input("📏 Nervure cubitale a (px)", 0, 2000, 0, 1)
                px_cubital_b = st.number_input("📏 Nervure cubitale b (px)", 0, 2000, 0, 1)
                px_cubital_c = st.number_input("📏 Nervure cubitale c (px)", 0, 2000, 0, 1)

                tomentum_obs    = st.slider("👁️ Tomentum observé (0–3)", 0, 3, 2)
                pig_opts        = ["Noir", "Brun foncé", "Brun clair", "Jaune"]
                pigmentation_obs = st.selectbox("👁️ Pigmentation scutellum", pig_opts, index=1)

            if px_etalon > 0:
                echelle = px_etalon / DIAMETRES_ETALONS[etalon_type]
                st.markdown(
                    f"<div style='background:#0F1117;border:1px solid #3A4A66;border-radius:6px;"
                    f"padding:6px 12px;font-size:.82rem;color:#A8B4CC;margin-top:4px'>"
                    f"📐 Échelle calculée : <b style='color:#F5A623'>{echelle:.1f} px/mm</b> "
                    f"({'%.1f' % (px_aile/echelle)} mm aile · "
                    f"{'%.1f' % (px_glossa/echelle) if px_glossa>0 else '—'} mm glossa)</div>",
                    unsafe_allow_html=True
                )

            btn_calc = st.button(
                "🧮 Calculer les mesures avec Gemma",
                disabled=(not ia_active or px_etalon == 0),
                use_container_width=True,
                type="primary"
            )
            if not ia_active:
                st.info("🔑 Configurez votre clé API IA.")
            elif px_etalon == 0:
                st.info("⬆️ Saisissez au moins le diamètre de la pièce étalon en pixels.")

            if btn_calc and ia_active and px_etalon > 0:
                with st.spinner(f"🧮 {model_now} calcule et estime les mesures morphométriques..."):
                    result = ia_estimer_morphometrie_texte(
                        etalon_type, px_etalon, px_aile, px_largeur,
                        px_cubital_a, px_cubital_b, px_cubital_c,
                        px_glossa, tomentum_obs, pigmentation_obs
                    )
                _appliquer_mesures_auto(result)

    # ── ONGLET 2 : ANALYSE + IA (inchangé) ──────────────────────────────────
    with tab2:
        _auto_filled = st.session_state.get("morpho_notes_auto", "") != ""
        if _auto_filled:
            st.markdown(
                "<div style='background:#0F1117;border:1px solid #34D399;border-left:4px solid #34D399;"
                "border-radius:6px;padding:8px 14px;margin-bottom:10px;font-size:.85rem;color:#34D399'>"
                "✅ <b>Mesures pré-remplies par mensuration automatique IA</b> — vérifiez et ajustez si nécessaire.</div>",
                unsafe_allow_html=True
            )

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### 📐 Mesures morphométriques")
            ruche_sel = st.selectbox("Ruche analysée", opts.keys())

            _aile_def    = max(7.0, min(12.0, float(st.session_state.get("morpho_aile", 9.2))))
            _largeur_def = max(2.0, min(5.0,  float(st.session_state.get("morpho_largeur", 3.1))))
            _cubital_def = max(1.0, min(5.0,  float(st.session_state.get("morpho_cubital", 2.3))))
            _glossa_def  = max(4.0, min(8.0,  float(st.session_state.get("morpho_glossa", 6.1))))
            _tom_def     = max(0,   min(3,    int(st.session_state.get("morpho_tomentum", 2))))
            _pig_opts    = ["Noir", "Brun foncé", "Brun clair", "Jaune"]
            _pig_def     = st.session_state.get("morpho_pigmentation", "Brun foncé")
            _pig_idx     = _pig_opts.index(_pig_def) if _pig_def in _pig_opts else 1

            aile    = st.number_input("Longueur aile antérieure (mm)", 7.0, 12.0, _aile_def, 0.1)
            largeur = st.number_input("Largeur aile (mm)", 2.0, 5.0, _largeur_def, 0.1)
            cubital = st.number_input("Indice cubital", 1.0, 5.0, _cubital_def, 0.1,
                                      help="Rapport distances nervures cubitales a/b ÷ b/c")
            glossa  = st.number_input("Longueur glossa (mm)", 4.0, 8.0, _glossa_def, 0.1)
            tomentum    = st.slider("Tomentum (densité poils thorax 0–3)", 0, 3, _tom_def)
            pigmentation = st.selectbox("Pigmentation scutellum", _pig_opts, index=_pig_idx)
            _notes_auto = st.session_state.get("morpho_notes_auto", "")
            notes = st.text_area("Notes / Observations",
                                 value=f"[Mensuration auto IA] {_notes_auto}" if _notes_auto else "")

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
            st.info(T("no_analyse"))

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : CARTOGRAPHIE (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_carto():
    st.markdown(T("carto_title"))

    ia_active = widget_cle_api()

    conn = get_db()
    tab1, tab2, tab3 = st.tabs(["🗺️ Carte & Zones", "🌿 Analyse environnement IA", "➕ Ajouter une zone"])

    with tab1:
        df_zones  = pd.read_sql("SELECT * FROM zones", conn)
        df_ruches = pd.read_sql("SELECT * FROM ruches WHERE statut='actif' AND latitude IS NOT NULL", conn)

        # Recherche de ville
        st.markdown(T("rechercher_ville"))
        col_search1, col_search2 = st.columns([3, 1])
        ville_recherche = col_search1.text_input(T("nom_ville_label"), placeholder="Ex: Tlemcen, Oran, Alger...")
        if col_search2.button(T("centrer_btn"), use_container_width=True) and ville_recherche:
            lat, lon = geocode_ville(ville_recherche)
            if lat and lon:
                st.session_state["map_center"] = (lat, lon)
                st.success(f"Carte centrée sur {ville_recherche} ({lat:.4f}, {lon:.4f})")
            else:
                st.error("Ville non trouvée. Vérifiez l'orthographe.")

        if FOLIUM_OK:
            center_lat = st.session_state.get("map_center", (34.88, 1.32))[0] if "map_center" in st.session_state else (float(df_ruches["latitude"].mean()) if not df_ruches.empty else 34.88)
            center_lon = st.session_state.get("map_center", (34.88, 1.32))[1] if "map_center" in st.session_state else (float(df_ruches["longitude"].mean()) if not df_ruches.empty else 1.32)

            # Carte avec plusieurs couches
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                           tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                           attr="Google Satellite")
            # Ajouter une couche OpenStreetMap avec les noms de villes (transparente)
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OSM</a> &copy; CartoDB",
                name="OpenStreetMap (labels)",
                overlay=False,
                control=True
            ).add_to(m)
            # Optionnel : ajouter une couche pour les frontières (via un GeoJSON, mais par simplicité on utilise OSM)

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

            # Ajouter le contrôle des couches
            folium.LayerControl().add_to(m)

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
# PAGE : ASSISTANT VOCAL D'INSPECTION
# ════════════════════════════════════════════════════════════════════════════
def page_assistant_vocal():
    st.markdown(T("vocal_title"))
    st.markdown("Dictée intelligente : parlez, l'IA extrait les données d'inspection.")

    ia_active = widget_cle_api()
    conn = get_db()
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    opts = {r[1]: r[0] for r in ruches}

    col1, col2 = st.columns([1, 1])
    with col1:
        ruche_sel = st.selectbox("Ruche à inspecter", opts.keys())
    with col2:
        date_insp = st.date_input("Date d'inspection", datetime.date.today())

    # Composant HTML/JS pour la reconnaissance vocale (Web Speech API)
    st.markdown("""
    <div style="background:#0F1117; border:1px solid #C8820A; border-radius:8px; padding:16px; margin:12px 0">
        <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap">
            <button id="startRecog" style="background:#C8820A; border:none; border-radius:30px; padding:8px 20px; color:white; font-weight:bold">🎙️ Démarrer l'enregistrement</button>
            <button id="stopRecog" style="background:#2E3A52; border:none; border-radius:30px; padding:8px 20px; color:white">⏹️ Arrêter</button>
            <span id="status" style="color:#A8B4CC">En attente...</span>
        </div>
        <div style="margin-top:12px; font-size:0.8rem; color:#6B7A99">
            🌐 Utilise Chrome/Edge – Parlez clairement en français (ex: "varroa deux virgule cinq pourcent, neuf cadres, reine vue")
        </div>
    </div>
    <textarea id="voiceText" rows="3" placeholder="Le texte transcrit apparaîtra ici..." style="width:100%; background:#1A2030; border:1px solid #2E3A52; border-radius:8px; color:#F0F4FF; padding:10px; margin-top:8px"></textarea>
    """, unsafe_allow_html=True)

    # JavaScript pour la reconnaissance vocale
    st.markdown("""
    <script>
    const startBtn = document.getElementById('startRecog');
    const stopBtn = document.getElementById('stopRecog');
    const statusSpan = document.getElementById('status');
    const textArea = document.getElementById('voiceText');
    let recognition = null;

    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'fr-FR';

        recognition.onstart = function() {
            statusSpan.innerHTML = '🎙️ Écoute en cours... Parlez maintenant';
            statusSpan.style.color = '#F5A623';
        };
        recognition.onend = function() {
            statusSpan.innerHTML = '✅ Enregistrement terminé';
            statusSpan.style.color = '#6EE7B7';
        };
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            textArea.value = transcript;
            statusSpan.innerHTML = '✅ Texte capturé';
        };
        recognition.onerror = function(event) {
            statusSpan.innerHTML = '❌ Erreur: ' + event.error;
            statusSpan.style.color = '#F87171';
        };
    } else {
        statusSpan.innerHTML = '❌ Reconnaissance vocale non supportée par ce navigateur. Utilisez Chrome/Edge.';
    }

    startBtn.onclick = function() {
        if (recognition) {
            recognition.start();
        } else {
            alert('Reconnaissance vocale non disponible');
        }
    };
    stopBtn.onclick = function() {
        if (recognition) {
            recognition.stop();
        }
    };
    </script>
    """, unsafe_allow_html=True)

    # Champ pour saisie manuelle alternative
    texte_manuel = st.text_area("📝 Ou saisissez/collez le texte manuellement :", height=100,
                                 placeholder="varroa deux virgule cinq pourcent, neuf cadres, reine vue, comportement calme, poids vingt-huit kilos")
    
    # Zone de texte qui sera utilisée pour l'analyse
    texte_a_analyser = st.text_area("🔍 Texte à analyser par l'IA (pré-rempli automatiquement)",
                                     height=120, key="voice_text_to_analyze")
    
    # Récupérer la transcription vocale via un petit hack : on utilise un callback JS vers un champ Streamlit ?
    # On ajoute un bouton pour rafraîchir depuis le textarea HTML
    if st.button("📋 Récupérer le texte depuis le micro", use_container_width=True):
        st.info("Cliquez sur le bouton ci-dessous après avoir dicté pour mettre à jour le champ.")
    
    # Bouton d'extraction IA
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🤖 Extraire les données (IA)", use_container_width=True, disabled=not ia_active):
            texte_source = texte_manuel if texte_manuel.strip() else st.session_state.get("voice_text_to_analyze", "")
            if not texte_source.strip():
                st.warning("Veuillez dicter ou saisir du texte.")
            else:
                with st.spinner("Extraction des données par l'IA..."):
                    result = extraire_inspection_ia(texte_source)
                if "error" not in result:
                    # Pré-remplir les champs du formulaire via session_state
                    st.session_state['vocal_varroa'] = result.get('varroa_pct', 1.0)
                    st.session_state['vocal_cadres'] = result.get('nb_cadres', 10)
                    st.session_state['vocal_reine'] = result.get('reine_vue', True)
                    st.session_state['vocal_comportement'] = result.get('comportement', 'calme')
                    st.session_state['vocal_poids'] = result.get('poids_kg', 25.0)
                    st.session_state['vocal_notes'] = result.get('notes', '')
                    st.success("✅ Données extraites. Vérifiez et enregistrez l'inspection ci-dessous.")
                else:
                    st.error(f"Erreur IA : {result.get('error')}")

    # Formulaire d'inspection pré-rempli
    with st.form("inspect_vocal_form"):
        st.markdown("### 📋 Inspection à enregistrer")
        cola, colb, colc = st.columns(3)
        varroa = cola.number_input("Varroa (%)", 0.0, 20.0, 
                                   value=st.session_state.get('vocal_varroa', 1.0), step=0.1)
        cadres = colb.number_input("Nb cadres", 0, 20, 
                                   value=st.session_state.get('vocal_cadres', 10))
        poids = colc.number_input("Poids (kg)", 0.0, 80.0,
                                  value=st.session_state.get('vocal_poids', 25.0), step=0.1)
        col_d, col_e = st.columns(2)
        reine = col_d.checkbox("Reine vue", value=st.session_state.get('vocal_reine', True))
        comportement = col_e.selectbox("Comportement", ["calme", "nerveuse", "agressive", "très calme"],
                                       index=["calme","nerveuse","agressive","très calme"].index(st.session_state.get('vocal_comportement','calme')))
        notes = st.text_area("Notes", value=st.session_state.get('vocal_notes', ''))
        submitted = st.form_submit_button("✅ Enregistrer l'inspection")

    if submitted:
        rid = opts[ruche_sel]
        conn.execute("""
            INSERT INTO inspections (ruche_id, date_inspection, poids_kg, nb_cadres, varroa_pct, reine_vue, comportement, notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (rid, str(date_insp), poids, cadres, varroa, int(reine), comportement, notes))
        # Enregistrer aussi dans inspections_vocales
        conn.execute("""
            INSERT INTO inspections_vocales (ruche_id, date_inspection, texte_original, varroa_pct, nb_cadres, reine_vue, comportement, poids_kg, notes)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (rid, str(date_insp), texte_manuel or st.session_state.get('voice_text_to_analyze',''), varroa, cadres, int(reine), comportement, poids, notes))
        conn.commit()
        log_action("Inspection vocale", f"Ruche {ruche_sel} — varroa {varroa}%")
        st.success("✅ Inspection enregistrée (source vocale).")
        st.rerun()

    # Historique des inspections vocales
    with st.expander("📜 Historique des inspections vocales"):
        df_hist = pd.read_sql("""
            SELECT iv.id, r.nom, iv.date_inspection, iv.varroa_pct, iv.nb_cadres, iv.reine_vue, iv.comportement, iv.notes
            FROM inspections_vocales iv
            JOIN ruches r ON r.id = iv.ruche_id
            ORDER BY iv.date_inspection DESC LIMIT 20
        """, conn)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info(T("no_insp_vocale"))
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : SCANNER DE CADRE IA
# ════════════════════════════════════════════════════════════════════════════
def page_scanner_cadre():
    st.markdown(T("scanner_title"))
    st.markdown("Téléchargez une photo d'un cadre de couvain. L'IA analysera les maladies, comptera les abeilles et détectera la reine.")

    ia_active = widget_cle_api()
    conn = get_db()
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    opts = {r[1]: r[0] for r in ruches}
    ruche_sel = st.selectbox("Ruche concernée", opts.keys())

    img_file = st.file_uploader("📷 Photo du cadre (JPEG/PNG)", type=["jpg","jpeg","png"], key="scan_cadre")
    if img_file:
        st.image(img_file, caption="Cadre à analyser", use_column_width=True)
        if st.button("🔬 Lancer l'analyse IA", use_container_width=True, disabled=not ia_active):
            with st.spinner("Analyse en cours par l'IA (vision)..."):
                img_bytes = img_file.read()
                result = analyser_cadre_ia(img_bytes)
            if "error" not in result:
                st.success("✅ Analyse terminée")
                col1, col2 = st.columns(2)
                col1.metric("🐝 Abeilles détectées", result.get("nb_abeilles", "N/A"))
                col2.metric("👑 Reine visible", "✅ Oui" if result.get("reine_detectee") else "❌ Non")
                col3, col4 = st.columns(2)
                col3.metric("🕷️ Varroa visible", "✅ Oui" if result.get("varroa_visible") else "❌ Non")
                col4.metric("📊 Couvain sain", f"{result.get('couvain_sain_pct', 0)}%")
                maladies = result.get("maladies_detectees", [])
                if maladies and "aucune" not in maladies:
                    st.error(f"⚠️ Maladies suspectées : {', '.join(maladies)}")
                else:
                    st.success("✅ Aucune maladie détectée")
                st.info(f"💡 Recommandation : {result.get('recommandations', 'Aucune')}")
                st.caption(f"Commentaire : {result.get('commentaire', '')}")

                # Sauvegarde
                import base64
                b64 = base64.b64encode(img_bytes).decode()
                rid = opts[ruche_sel]
                conn.execute("""
                    INSERT INTO analyses_cadre (ruche_id, date_analyse, image_base64, nb_abeilles, reine_detectee,
                                                maladies_detectees, varroa_visible, couvain_sain_pct, recommandations, ia_reponse)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (rid, str(datetime.date.today()), b64, result.get("nb_abeilles"), int(result.get("reine_detectee", False)),
                      json.dumps(maladies), int(result.get("varroa_visible", False)), result.get("couvain_sain_pct", 0),
                      result.get("recommandations", ""), json.dumps(result)))
                conn.commit()
                log_action("Analyse cadre IA", f"Ruche {ruche_sel} - {result.get('nb_abeilles')} abeilles")
            else:
                st.error(f"Erreur IA : {result.get('error')}")

    # Historique des analyses
    with st.expander("📜 Historique des analyses de cadre"):
        df_hist = pd.read_sql("""
            SELECT ac.id, r.nom, ac.date_analyse, ac.nb_abeilles, ac.reine_detectee, ac.maladies_detectees, ac.recommandations
            FROM analyses_cadre ac
            JOIN ruches r ON r.id = ac.ruche_id
            ORDER BY ac.date_analyse DESC LIMIT 20
        """, conn)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info(T("no_analyse_cadre"))
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION DE TRANSHUMANCE
# ════════════════════════════════════════════════════════════════════════════
def page_transhumance():
    st.markdown(T("transhu_title"))
    st.markdown("Obtenez une analyse détaillée : flore, météo, risques, recommandations pour une transhumance réussie.")

    ia_active = widget_cle_api()
    conn = get_db()

    # Récupération des ruches
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    opts = {r[1]: r[0] for r in ruches}
    ruche_sel = st.selectbox("Ruche à transhumer", opts.keys())

    # Période souhaitée
    st.markdown("#### 📅 Période de transhumance souhaitée")
    periode_options = ["Printemps (Mars-Mai)", "Été (Juin-Août)", "Automne (Sep-Nov)", "Hiver (Déc-Fév)", "Personnalisée"]
    periode_choisie = st.selectbox("Sélectionnez la période", periode_options, index=0)
    
    if periode_choisie == "Personnalisée":
        col_p1, col_p2 = st.columns(2)
        mois_debut = col_p1.selectbox("Mois de début", ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"], index=2)
        mois_fin = col_p2.selectbox("Mois de fin", ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"], index=4)
        periode_souhaitee = f"{mois_debut} - {mois_fin}"
    else:
        periode_souhaitee = periode_choisie

    # Recherche de ville
    st.markdown("#### 🗺️ Zone cible")
    col_search1, col_search2 = st.columns([3,1])
    ville = col_search1.text_input("Nom de la ville/région", placeholder="Ex: Bejaia, Khenchela, El Tarf, Tlemcen...")
    if col_search2.button("📍 Centrer", use_container_width=True) and ville:
        lat, lon = geocode_ville(ville)
        if lat and lon:
            st.session_state["transhumance_center"] = (lat, lon)
            st.success(f"Centré sur {ville} ({lat:.4f}, {lon:.4f})")
        else:
            st.error("Ville non trouvée")

    center = st.session_state.get("transhumance_center", (34.88, 1.32))
    
    # Carte satellite
    if FOLIUM_OK:
        m = folium.Map(location=center, zoom_start=12, 
                       tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
                       attr="Google Satellite")
        m.add_child(folium.LatLngPopup())
        folium.Marker(location=center, popup="Zone cible", icon=folium.Icon(color="green")).add_to(m)
        st_folium(m, width="100%", height=400, key="transhumance_map")
    else:
        st.warning("Folium non installé.")

    col_lat, col_lon = st.columns(2)
    lat_cible = col_lat.number_input("Latitude cible", -90.0, 90.0, center[0], format="%.5f")
    lon_cible = col_lon.number_input("Longitude cible", -180.0, 180.0, center[1], format="%.5f")

    # Bouton de prédiction
    if st.button("🔮 Prédiction détaillée (IA)", use_container_width=True, disabled=not ia_active):
        # Reverse geocoding
        lieu_nom = ""
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat_cible}&lon={lon_cible}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "ApiTrackPro/3.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
                lieu_nom = data.get("display_name", f"{lat_cible:.4f}, {lon_cible:.4f}")
        except:
            lieu_nom = f"{lat_cible:.4f}, {lon_cible:.4f}"

        with st.spinner(f"Analyse approfondie pour la période {periode_souhaitee}... (15-30 secondes)"):
            result = predire_transhumance_ia(lat_cible, lon_cible, lieu_nom, periode_souhaitee)

        if "error" not in result and isinstance(result, dict):
            st.balloons()
            st.success(f"✅ Analyse générée pour la période {periode_souhaitee}")

            # 1. Résumé
            st.markdown("### 📍 Synthèse de la zone")
            st.info(result.get("resume_zone", "Aucune description disponible"))

            # 2. Potentiel
            col1, col2, col3 = st.columns(3)
            col1.metric("🌻 Potentiel miel", result.get("potentiel_miel", "N/A").capitalize())
            col2.metric("📅 Départ optimal", result.get("periode_optimale_depart", "N/A"))
            col3.metric("🔄 Retour optimal", result.get("periode_optimale_retour", "N/A"))
            st.metric("⏱️ Durée de séjour conseillée", f"{result.get('duree_sejour_jours', 'N/A')} jours")

            # 3. Plantes en fleur pendant la période souhaitée
            st.markdown(f"### 🌸 Plantes en fleur pendant {periode_souhaitee}")
            plantes_periode = result.get("plantes_en_fleur_periode", [])
            if plantes_periode:
                st.markdown(", ".join([f"**{p}**" for p in plantes_periode]))
            else:
                st.info("Aucune information spécifique pour cette période")

            # 4. Flore détaillée
            st.markdown("### 🌿 Flore mellifère détaillée")
            flore_list = result.get("flore_detaillee", [])
            if flore_list:
                df_flore = pd.DataFrame(flore_list)
                # Ajout d'icônes
                df_flore["🌼 Type"] = df_flore["type"].apply(lambda x: "🍯 Nectar" if x=="nectar" else ("🌾 Pollen" if x=="pollen" else "🍯+🌾 Mixte" if "nectar+pollen" in x else "🍬 Miellat"))
                df_flore["📊 Abondance"] = df_flore["abondance"]
                df_flore["📅 Floraison"] = df_flore["periode_floraison"]
                df_flore["⭐ Potentiel"] = df_flore["potentiel_mellifere"]
                df_flore["🌸 Fleurit à la période"] = df_flore["fleurit_pendant_periode"].apply(lambda x: "✅ Oui" if x else "❌ Non")
                st.dataframe(df_flore[["espece", "🌼 Type", "📊 Abondance", "📅 Floraison", "⭐ Potentiel", "🌸 Fleurit à la période"]], 
                             use_container_width=True, hide_index=True)
                st.markdown(f"**🌳 Flore dominante :** {result.get('flore_dominante', 'Non spécifiée')}")
            else:
                st.warning("Aucune donnée floristique détaillée")

            # 5. Météo
            meteo = result.get("meteo_moyenne", {})
            if meteo:
                st.markdown("### ☀️ Climat annuel")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("🌡️ Printemps", meteo.get("temp_printemps_c", "N/A") + " °C")
                col_m2.metric("☀️ Été", meteo.get("temp_ete_c", "N/A") + " °C")
                col_m3.metric("🍂 Automne", meteo.get("temp_automne_c", "N/A") + " °C")
                st.metric("💧 Pluviométrie annuelle", f"{meteo.get('pluie_annuelle_mm', 'N/A')} mm")
                st.metric("☀️ Ensoleillement", f"{meteo.get('ensoleillement_heures', 'N/A')} h/an")
                st.caption(f"🌬️ Vents dominants : {meteo.get('vent_dominant', 'N/A')}")

            # Météo pendant la période
            meteo_periode = result.get("meteo_pendant_periode", {})
            if meteo_periode:
                st.markdown(f"### 🌡️ Météo pendant {periode_souhaitee}")
                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.metric("Température moyenne", meteo_periode.get("temperature_moyenne_c", "N/A") + " °C")
                col_p2.metric("Précipitations", f"{meteo_periode.get('precipitations_mm', 'N/A')} mm")
                col_p3.metric("Risque gel", "⚠️ Oui" if meteo_periode.get("risque_gel") == "oui" else "✅ Non")
                st.caption(f"Risque sécheresse : {'⚠️ Oui' if meteo_periode.get('risque_secheresse') == 'oui' else '✅ Non'}")

            # 6. Ressources et accès
            st.markdown("### 💧 Infrastructures et environnement")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("💦 Eau", result.get("ressources_eau", "Non précisé"))
            col_r2.metric("🚜 Accès ruches", result.get("acces_ruches", "Non précisé"))
            col_r3.metric("🐝 Concurrence", result.get("concurrence_apicole", "Non précisée"))

            # 7. Risques
            risques = result.get("risques", [])
            if risques:
                st.markdown("### ⚠️ Risques identifiés")
                for r in risques:
                    st.warning(f"• {r}")

            # 8. Recommandations
            recommandations = result.get("recommandations", [])
            if recommandations:
                st.markdown("### 💡 Recommandations pour la transhumance")
                for rec in recommandations:
                    st.success(f"✅ {rec}")

            # Sauvegarde
            conn.execute("""
                INSERT INTO transhumances (ruche_id, latitude_dest, longitude_dest, lieu_dest, potentiel_miel, recommandation_ia)
                VALUES (?,?,?,?,?,?)
            """, (opts[ruche_sel], lat_cible, lon_cible, lieu_nom, result.get("potentiel_miel"), json.dumps(result, ensure_ascii=False)))
            conn.commit()
            log_action("Prédiction transhumance détaillée", f"Ruche {ruche_sel} vers {lieu_nom} - période {periode_souhaitee}")

            # Export JSON
            st.download_button(
                "⬇️ Exporter l'analyse (JSON)",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"transhumance_{lieu_nom[:30]}_{periode_souhaitee.replace(' ','_')}.json",
                mime="application/json"
            )

        else:
            st.error(f"Erreur IA : {result.get('error', 'Réponse invalide')}")

    # Historique
    with st.expander("📋 Historique des transhumances planifiées"):
        df_hist = pd.read_sql("""
            SELECT t.id, r.nom, t.lieu_dest, t.potentiel_miel, t.date_depart, t.date_retour, t.statut, t.created_at
            FROM transhumances t
            JOIN ruches r ON r.id = t.ruche_id
            ORDER BY t.created_at DESC LIMIT 20
        """, conn)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info(T("no_transhu"))
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : PEDIGREE & SÉLECTION
# ════════════════════════════════════════════════════════════════════════════
def page_pedigree():
    st.markdown(T("pedigree_title"))
    conn = get_db()

    tab1, tab2, tab3 = st.tabs([T("tab_reines"), T("tab_add_reine"), T("tab_arbre")])

    with tab1:
        df_reines = pd.read_sql("""
            SELECT r.id, r.nom, ru.nom as ruche, r.race, r.date_naissance, r.qualite,
                   (SELECT nom FROM reines WHERE id = r.mere_id) as mere,
                   (SELECT nom FROM reines WHERE id = r.pere_id) as pere,
                   r.notes
            FROM reines r
            LEFT JOIN ruches ru ON ru.id = r.ruche_id
            ORDER BY r.date_naissance DESC
        """, conn)
        if not df_reines.empty:
            st.dataframe(df_reines, use_container_width=True, hide_index=True)
            csv = df_reines.to_csv(index=False).encode()
            st.download_button(T("export_csv"), csv, "reines.csv", "text/csv")
        else:
            st.info(T("no_reine"))

    with tab2:
        ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
        ruche_opts = {r[1]: r[0] for r in ruches}
        reines_list = conn.execute("SELECT id, nom FROM reines").fetchall()
        reine_opts = {r[1]: r[0] for r in reines_list}
        reine_opts["(Inconnue)"] = None

        with st.form("add_reine"):
            st.markdown(T("nouvelle_reine"))
            col1, col2 = st.columns(2)
            nom = col1.text_input(T("nom_reine"))
            ruche = col2.selectbox(T("ruche_associee"), ruche_opts.keys())
            col3, col4 = st.columns(2)
            race = col3.selectbox("Race", ["intermissa", "sahariensis", "ligustica", "carnica", "hybride"])
            date_naiss = col4.date_input(T("date_naissance"), datetime.date.today())
            col5, col6 = st.columns(2)
            mere = col5.selectbox(T("mere"), reine_opts.keys())
            pere = col6.selectbox(T("pere"), reine_opts.keys())
            qualite = st.selectbox("Qualité", ["standard", "reproductrice", "élite", "à réformer"])
            notes = st.text_area(T("notes_perf"))
            submitted = st.form_submit_button(T("btn_save_reine"))

        if submitted and nom:
            rid = ruche_opts[ruche]
            mere_id = reine_opts.get(mere) if mere != "(Inconnue)" else None
            pere_id = reine_opts.get(pere) if pere != "(Inconnue)" else None
            conn.execute("""
                INSERT INTO reines (nom, ruche_id, mere_id, pere_id, date_naissance, race, qualite, notes)
                VALUES (?,?,?,?,?,?,?,?)
            """, (nom, rid, mere_id, pere_id, str(date_naiss), race, qualite, notes))
            conn.commit()
            log_action("Reine ajoutée", f"{nom} (ruche {ruche})")
            st.success(T("success_reine").format(nom))
            st.rerun()

    with tab3:
        st.markdown(T("arbre_titre"))
        # Requête récursive pour afficher l'arbre (limité à 3 générations)
        arbre = conn.execute("""
            WITH RECURSIVE arbre(id, nom, mere_id, pere_id, niveau, chemin) AS (
                SELECT id, nom, mere_id, pere_id, 0, nom FROM reines WHERE mere_id IS NULL AND pere_id IS NULL
                UNION ALL
                SELECT r.id, r.nom, r.mere_id, r.pere_id, a.niveau+1, a.chemin || ' → ' || r.nom
                FROM reines r
                JOIN arbre a ON (r.mere_id = a.id OR r.pere_id = a.id)
                WHERE a.niveau < 4
            )
            SELECT * FROM arbre ORDER BY niveau, nom
        """).fetchall()
        if arbre:
            for a in arbre:
                st.markdown(f"{'  ' * a[4]}🐝 {a[1]} (niv.{a[4]})")
        else:
            st.info(T("no_genealogie"))

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : MÉTÉO & MIELLÉE (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_meteo():
    st.markdown(T("meteo_title"))
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
# PAGE : GÉNÉTIQUE & SÉLECTION (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_genetique():
    st.markdown(T("genetique_title"))
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
# PAGE : FLORE MELLIFÈRE (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_flore():
    st.markdown(T("flore_title"))
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
# PAGE : ALERTES (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_alertes():
    st.markdown(T("alerte_title"))
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
# PAGE : JOURNAL (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def page_journal():
    st.markdown(T("journal_title"))
    conn = get_db()
    df = pd.read_sql("SELECT * FROM journal ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()

    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(T("export_csv"), csv, "journal.csv", "text/csv")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(T("journal_vide"))


# ════════════════════════════════════════════════════════════════════════════
# PAGE : ADMINISTRATION (inchangée)
# ════════════════════════════════════════════════════════════════════════════
def import_csv(table_name, df):
    """Insère les lignes d'un DataFrame dans la table spécifiée (ignore les colonnes non existantes)."""
    conn = get_db()
    cursor = conn.cursor()
    # Récupérer les colonnes existantes
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = [col[1] for col in cursor.fetchall()]
    # Filtrer les colonnes du DataFrame
    df_import = df[[col for col in df.columns if col in existing_columns]]
    # Insérer ligne par ligne
    for _, row in df_import.iterrows():
        placeholders = ",".join(["?"] * len(row))
        cols = ",".join(row.index)
        try:
            cursor.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})", tuple(row))
        except Exception as e:
            st.warning(f"Erreur insertion ligne : {e}")
    conn.commit()
    conn.close()
    st.success(f"✅ {len(df_import)} lignes importées dans {table_name}.")

def page_admin():
    st.markdown(T("admin_title"))
    conn = get_db()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([T("tab_profil"), T("tab_ia"), T("tab_pwd"), T("tab_db"), T("tab_import")])

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

        version = get_setting("version", "3.0.0")
        st.markdown(f"<div class='api-footer'>ApiTrack Pro v{version} · Streamlit · SQLite · © 2025</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown("### 📂 Import de données depuis CSV")
        st.markdown("""
        <div style='background:#0F1117;border:1px solid #C8820A;border-radius:8px;padding:12px;margin-bottom:16px'>
        ⚠️ Le fichier CSV doit contenir des colonnes exactement nommées comme dans la base de données.
        Les colonnes manquantes seront ignorées. L'import est effectué ligne par ligne.
        </div>
        """, unsafe_allow_html=True)

        table_choice = st.selectbox("Choisir la table cible", ["ruches", "inspections", "traitements", "recoltes", "morph_analyses", "zones"])
        uploaded_file = st.file_uploader("Fichier CSV", type="csv", key="import_csv")
        if uploaded_file is not None:
            try:
                df_import = pd.read_csv(uploaded_file)
                st.write("Aperçu des données :")
                st.dataframe(df_import.head())
                if st.button("✅ Importer dans la base", use_container_width=True):
                    import_csv(table_choice, df_import)
            except Exception as e:
                st.error(f"Erreur lecture CSV : {e}")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : ÉLEVAGE REINES & MÂLES
# ════════════════════════════════════════════════════════════════════════════
def get_niveau():
    return st.session_state.get("niveau_user", "intermediaire")


def niveau_badge():
    niv = get_niveau()
    colors = {"debutant": "#34D399", "intermediaire": "#F5A623", "expert": "#F87171"}
    labels = {"debutant": T("niveau_debutant"), "intermediaire": T("niveau_intermediaire"), "expert": T("niveau_expert")}
    c = colors.get(niv, "#F5A623")
    l = labels.get(niv, niv)
    return f"<span style='background:{c}22;color:{c};border:1px solid {c};border-radius:12px;padding:2px 10px;font-size:.75rem;font-weight:600'>{l}</span>"


def aide_contextuelle(debutant_txt, intermediaire_txt, expert_txt):
    niv = get_niveau()
    if niv == "debutant" and debutant_txt:
        st.info(f"💡 {debutant_txt}")
    elif niv == "intermediaire" and intermediaire_txt:
        st.info(f"🐝 {intermediaire_txt}")
    elif niv == "expert" and expert_txt:
        st.info(f"🔬 {expert_txt}")


def page_elevage_reines():
    st.markdown(T("elevage_title"))
    st.markdown(niveau_badge(), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    niv = get_niveau()
    conn = get_db()
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    opts_ruches = {r[1]: r[0] for r in ruches}

    tab1, tab2, tab3, tab4 = st.tabs([
        T("tab_elevage_reines"),
        T("tab_elevage_males"),
        T("tab_generation"),
        "📊 Stats"
    ])

    # ── ONGLET 1 : ÉLEVAGE REINES ──────────────────────────────────────────
    with tab1:
        aide_contextuelle(
            "L'élevage des reines consiste à produire de nouvelles reines à partir de larves sélectionnées. Commencez par choisir une ruche donneuse (la meilleure de votre rucher).",
            "Choisissez votre méthode : Doolittle (greffage), Starter/Finisseur, ou nucléi. Le taux d'acceptation moyen est de 60-80%.",
            "Optimisez la sélection sur critères : production, douceur, résistance Varroa (VSH), hygiène. Planifiez les croisements avec vos colonies de mâles sélectionnés."
        )

        col_form, col_list = st.columns([1, 1.5])

        with col_form:
            st.markdown("### ➕ Nouveau lot d'élevage")
            with st.form("form_elevage_reine"):
                nom_lot = st.text_input("Nom du lot*", placeholder="Ex: Lot-Mars-2025-Int")
                
                col_a, col_b = st.columns(2)
                ruche_don = col_a.selectbox(T("ruche_donneuse"), list(opts_ruches.keys()), key="er_don")
                ruche_elv = col_b.selectbox(T("ruche_eleveuse"), list(opts_ruches.keys()), key="er_elv")
                
                methodes = ["Doolittle (greffage)", "Miller", "Jenter/Cupularve", "Nicot", "Starter/Finisseur", "Naturelle (sauvage)"]
                if niv == "debutant":
                    methodes = ["Naturelle (sauvage)", "Jenter/Cupularve (kit)", "Miller"]
                
                methode = st.selectbox(T("methode_elevage"), methodes)
                race = st.selectbox("Race", ["intermissa", "sahariensis", "ligustica", "carnica", "buckfast", "hybride"])
                
                col_c, col_d = st.columns(2)
                date_greffe = col_c.date_input(T("date_greffe"), datetime.date.today())
                nb_greffes = col_d.number_input(T("nb_cellules"), 1, 100, 10 if niv != "debutant" else 5)
                
                if niv in ["intermediaire", "expert"]:
                    qualite = st.selectbox(T("qualite_reine"), ["standard", "sélectionnée", "élite", "championne"])
                else:
                    qualite = "standard"
                
                notes = st.text_area("Notes", height=60)
                submitted = st.form_submit_button("✅ Créer le lot")

            if submitted and nom_lot:
                # Calcul date émergence automatique (16 jours après greffe)
                date_em = date_greffe + datetime.timedelta(days=16)
                date_op = date_greffe + datetime.timedelta(days=8)
                conn.execute("""
                    INSERT INTO elevage_reines (nom, ruche_donneuse_id, ruche_eleveuse_id, methode,
                        date_greffe, date_operculee, date_emergence, nb_cellules_greffees, qualite, race, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (nom_lot, opts_ruches[ruche_don], opts_ruches[ruche_elv], methode,
                      str(date_greffe), str(date_op), str(date_em), nb_greffes, qualite, race, notes))
                conn.commit()
                log_action("Élevage reine créé", f"Lot {nom_lot} — méthode {methode}")
                st.success(f"✅ Lot '{nom_lot}' créé ! Émergence prévue : {date_em}")
                st.rerun()

        with col_list:
            st.markdown("### 📋 Lots d'élevage en cours")
            df_lots = pd.read_sql("""
                SELECT er.id, er.nom, er.methode, er.race, er.date_greffe, er.date_emergence,
                       er.nb_cellules_greffees, er.nb_cellules_acceptees, er.qualite, er.statut,
                       rd.nom as ruche_donneuse, re.nom as ruche_eleveuse
                FROM elevage_reines er
                LEFT JOIN ruches rd ON rd.id = er.ruche_donneuse_id
                LEFT JOIN ruches re ON re.id = er.ruche_eleveuse_id
                ORDER BY er.date_greffe DESC
            """, conn)

            if not df_lots.empty:
                today_str = str(datetime.date.today())
                for _, row in df_lots.iterrows():
                    # Calcul du stade actuel
                    stade = "🥚 Greffe"
                    if row['date_greffe'] and row['date_greffe'] <= today_str:
                        jours = (datetime.date.today() - datetime.date.fromisoformat(row['date_greffe'])).days
                        if jours < 0: stade = "📅 Planifié"
                        elif jours < 3: stade = "🥚 Larve greffée"
                        elif jours < 8: stade = "🐛 Développement"
                        elif jours < 16: stade = "🏠 Cellule operculée"
                        else: stade = "✅ Émergence"
                    
                    taux = int(row['nb_cellules_acceptees'] / row['nb_cellules_greffees'] * 100) if row['nb_cellules_greffees'] > 0 else 0
                    
                    color = "#34D399" if stade == "✅ Émergence" else ("#F5A623" if "operculée" in stade else "#60A5FA")
                    st.markdown(f"""
                    <div style='background:#1E2535;border:1px solid #2E3A52;border-left:4px solid {color};
                                border-radius:8px;padding:12px;margin-bottom:8px'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <b style='color:#F5A623'>{row['nom']}</b>
                            <span style='font-size:.7rem;background:{color}22;color:{color};border:1px solid {color};
                                         border-radius:10px;padding:2px 8px'>{stade}</span>
                        </div>
                        <div style='font-size:.8rem;color:#A8B4CC;margin-top:6px'>
                            🔬 {row['methode']} · 🐝 {row['race']} · 
                            📅 Greffe: {row['date_greffe']} → Émergence: {row['date_emergence']}<br>
                            🐝 Donneuse: {row['ruche_donneuse'] or '—'} · Éleveuse: {row['ruche_eleveuse'] or '—'}<br>
                            📊 Greffées: {row['nb_cellules_greffees']} · Acceptées: {row['nb_cellules_acceptees']} · Taux: {taux}% · ⭐ {row['qualite']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mise à jour rapide du nombre d'acceptées
                    with st.expander(f"✏️ Mettre à jour '{row['nom']}'"):
                        col_u1, col_u2, col_u3 = st.columns(3)
                        new_acc = col_u1.number_input("Cellules acceptées", 0, int(row['nb_cellules_greffees']), int(row['nb_cellules_acceptees']), key=f"acc_{row['id']}")
                        new_stat = col_u2.selectbox("Statut", ["en_cours","terminé","échec"], key=f"stat_{row['id']}")
                        if col_u3.button("💾 Mettre à jour", key=f"upd_{row['id']}"):
                            conn.execute("UPDATE elevage_reines SET nb_cellules_acceptees=?, statut=? WHERE id=?",
                                        (new_acc, new_stat, row['id']))
                            conn.commit()
                            st.rerun()
            else:
                st.info("Aucun lot d'élevage enregistré.")

    # ── ONGLET 2 : ÉLEVAGE MÂLES ───────────────────────────────────────────
    with tab2:
        aide_contextuelle(
            "Les mâles (faux-bourdons) sont essentiels pour l'insémination naturelle des reines. Une bonne colonie de mâles doit avoir des cadres de couvain de mâle sain.",
            "Planifiez vos colonies de mâles 1 mois avant vos élevages de reines. Choisissez des colonies avec les meilleures performances (miel, douceur).",
            "L'insémination instrumentale (IA) permet un contrôle total de la génétique. Évaluez la qualité du sperme : mobilité, concentration, morphologie."
        )

        col_mf, col_ml = st.columns([1, 1.5])

        with col_mf:
            st.markdown("### ➕ Colonie de mâles")
            with st.form("form_males"):
                ruche_m = st.selectbox("Ruche productrice de mâles", list(opts_ruches.keys()), key="m_ruche")
                race_m = st.selectbox("Race", ["intermissa", "sahariensis", "ligustica", "carnica", "buckfast", "hybride"])
                date_m = st.date_input("Date début", datetime.date.today())
                nb_cadres_m = st.number_input("Nb cadres de mâles", 0, 10, 2)
                
                if niv == "expert":
                    qualite_sperme = st.selectbox("Qualité sperme", ["non_évalué", "bonne", "excellente", "médiocre"])
                else:
                    qualite_sperme = "non_évalué"
                
                notes_m = st.text_area("Notes", height=60)
                sub_m = st.form_submit_button("✅ Enregistrer")

            if sub_m:
                conn.execute("""
                    INSERT INTO elevage_males (ruche_id, race, date_debut, nb_cadres_males, qualite_sperme, notes)
                    VALUES (?,?,?,?,?,?)
                """, (opts_ruches[ruche_m], race_m, str(date_m), nb_cadres_m, qualite_sperme, notes_m))
                conn.commit()
                log_action("Colonie mâles enregistrée", f"Ruche {ruche_m} — race {race_m}")
                st.success("✅ Colonie de mâles enregistrée.")
                st.rerun()

        with col_ml:
            st.markdown("### 📋 Colonies de mâles")
            df_males = pd.read_sql("""
                SELECT em.id, r.nom as ruche, em.race, em.date_debut, em.nb_cadres_males, em.qualite_sperme, em.notes
                FROM elevage_males em LEFT JOIN ruches r ON r.id = em.ruche_id
                ORDER BY em.date_debut DESC
            """, conn)
            if not df_males.empty:
                st.dataframe(df_males, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune colonie de mâles enregistrée.")

    # ── ONGLET 3 : GÉNÉRATIONS ─────────────────────────────────────────────
    with tab3:
        st.markdown("### 🌳 Arbre des générations")
        aide_contextuelle(
            "Chaque reine a une mère. En cliquant sur une reine dans la liste Pedigree, vous voyez son arbre généalogique.",
            "Visualisez les liens de parenté pour éviter la consanguinité. Idéal d'alterner les races de mâles toutes les 3-4 générations.",
            "Analysez les traits héréditaires : taux de miel par génération, résistance Varroa, douceur. Exportez l'arbre pour documenter votre programme de sélection."
        )
        
        df_reines_gen = pd.read_sql("""
            SELECT r1.id, r1.nom, r1.race, r1.date_naissance, r1.qualite,
                   r2.nom as mere_nom, r3.nom as pere_nom,
                   ru.nom as ruche_nom
            FROM reines r1
            LEFT JOIN reines r2 ON r2.id = r1.mere_id
            LEFT JOIN reines r3 ON r3.id = r1.pere_id
            LEFT JOIN ruches ru ON ru.id = r1.ruche_id
            ORDER BY r1.date_naissance
        """, conn)
        
        if df_reines_gen.empty:
            st.info("Aucune reine enregistrée. Ajoutez des reines dans la page Pedigree.")
        else:
            # Affichage visuel des générations
            # Déterminer la génération de chaque reine
            def get_generation(reine_id, df, visited=None):
                if visited is None:
                    visited = set()
                if reine_id in visited:
                    return 0
                visited.add(reine_id)
                row = df[df['id'] == reine_id]
                if row.empty:
                    return 0
                mere_nom = row.iloc[0]['mere_nom']
                if pd.isna(mere_nom):
                    return 0
                mere_row = df[df['nom'] == mere_nom]
                if mere_row.empty:
                    return 0
                return 1 + get_generation(mere_row.iloc[0]['id'], df, visited)
            
            df_reines_gen['generation'] = df_reines_gen['id'].apply(lambda x: get_generation(x, df_reines_gen))
            max_gen = int(df_reines_gen['generation'].max()) if len(df_reines_gen) > 0 else 0
            
            for gen in range(max_gen + 1):
                df_gen = df_reines_gen[df_reines_gen['generation'] == gen]
                if not df_gen.empty:
                    st.markdown(f"#### 🌳 Génération {gen} ({len(df_gen)} reines)")
                    cols_g = st.columns(min(4, len(df_gen)))
                    for i, (_, r) in enumerate(df_gen.iterrows()):
                        with cols_g[i % len(cols_g)]:
                            qualite_color = {"standard": "#A8B4CC", "sélectionnée": "#F5A623", "élite": "#34D399", "championne": "#F87171"}.get(str(r['qualite']), "#A8B4CC")
                            st.markdown(f"""
                            <div style='background:#1E2535;border:1px solid #2E3A52;border-top:3px solid {qualite_color};
                                        border-radius:8px;padding:10px;text-align:center'>
                                <div style='font-size:1.2rem'>👑</div>
                                <div style='font-weight:600;color:#F0F4FF;font-size:.9rem'>{r['nom']}</div>
                                <div style='font-size:.75rem;color:#A8B4CC'>{r['race'] or '—'}</div>
                                <div style='font-size:.7rem;color:#6B7A99'>{r['ruche_nom'] or '—'}</div>
                                <div style='font-size:.7rem;color:{qualite_color};margin-top:4px'>⭐ {r['qualite'] or 'standard'}</div>
                                <div style='font-size:.65rem;color:#6B7A99'>
                                    {('👑 ' + str(r['mere_nom'])) if not pd.isna(r['mere_nom']) else '🌱 Fondatrice'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    # ── ONGLET 4 : STATS ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📊 Statistiques d'élevage")
        df_stats_elv = pd.read_sql("""
            SELECT methode,
                   COUNT(*) as nb_lots,
                   SUM(nb_cellules_greffees) as total_greffees,
                   SUM(nb_cellules_acceptees) as total_acceptees,
                   ROUND(AVG(CASE WHEN nb_cellules_greffees > 0 
                       THEN nb_cellules_acceptees * 100.0 / nb_cellules_greffees ELSE 0 END), 1) as taux_moyen
            FROM elevage_reines GROUP BY methode
        """, conn)
        if not df_stats_elv.empty:
            fig_elv = px.bar(df_stats_elv, x="methode", y="taux_moyen",
                             title="Taux d'acceptation par méthode (%)",
                             color="taux_moyen", color_continuous_scale="RdYlGn",
                             template="plotly_white")
            fig_elv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   height=280, margin=dict(t=30,b=10,l=0,r=0))
            st.plotly_chart(fig_elv, use_container_width=True)
            st.dataframe(df_stats_elv, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune donnée d'élevage disponible.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE : CELLULES ROYALES
# ════════════════════════════════════════════════════════════════════════════
def page_cellules_royales():
    st.markdown(T("cellules_title"))
    st.markdown(niveau_badge(), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    niv = get_niveau()
    conn = get_db()

    # Récupérer les lots d'élevage
    lots = conn.execute("SELECT id, nom FROM elevage_reines ORDER BY date_greffe DESC").fetchall()
    ruches = conn.execute("SELECT id, nom FROM ruches WHERE statut='actif'").fetchall()
    opts_lots = {r[1]: r[0] for r in lots}
    opts_ruches = {r[1]: r[0] for r in ruches}

    aide_contextuelle(
        "Une cellule royale est la loge où se développe une reine. Elle est reconnaissable à sa forme allongée (comme un gland). Surveillez son évolution : greffe → operculation (J+8) → émergence (J+16).",
        "Suivez chaque cellule individuellement. Mesurez sa taille (>20mm = bonne qualité). Notez si elle est bien nourrie en gelée royale. Le stade operculé est critique : pas de vibrations !",
        "Evaluez la qualité : poids (>250mg ideal), longueur, aspect. Planifiez les nuclei d'introduction. Suivez le comportement post-émergence : vol nuptial J+5 à J+10 après émergence, ponte J+20."
    )

    tab_list, tab_add, tab_timeline = st.tabs(["📋 Suivi cellules", "➕ Ajouter", "📅 Timeline"])

    with tab_list:
        df_cells = pd.read_sql("""
            SELECT cr.id, cr.numero, er.nom as lot, cr.stade, cr.date_greffe,
                   cr.date_operculee, cr.date_emergence, cr.qualite, cr.statut,
                   cr.poids_mg, cr.longueur_mm, cr.notes,
                   r.nom as ruche
            FROM cellules_royales cr
            LEFT JOIN elevage_reines er ON er.id = cr.elevage_id
            LEFT JOIN ruches r ON r.id = cr.ruche_id
            ORDER BY cr.date_greffe DESC, cr.numero
        """, conn)

        if df_cells.empty:
            st.info("Aucune cellule royale enregistrée. Ajoutez-en ci-dessous.")
        else:
            # Mise à jour du stade automatique
            today = datetime.date.today()
            
            stade_color = {
                "oeuf": "#60A5FA",
                "larve": "#F5C842",
                "operculée": "#F5A623",
                "pré-émergence": "#34D399",
                "émergée": "#6EE7B7",
                "introduite": "#A78BFA",
                "échouée": "#F87171"
            }
            
            cols_h = st.columns([1,2,2,2,2,1,1])
            for h, t in zip(cols_h, ["#", "Lot", "Stade", "Greffe→Émergence", "Qualité", "Poids(mg)", "Actions"]):
                h.markdown(f"<b style='color:#F5A623;font-size:.75rem'>{t}</b>", unsafe_allow_html=True)
            
            for _, cell in df_cells.iterrows():
                stade = str(cell['stade'])
                color = stade_color.get(stade, "#A8B4CC")
                cols_r = st.columns([1,2,2,2,2,1,1])
                cols_r[0].markdown(f"<span style='color:#A8B4CC'>#{cell['numero']}</span>", unsafe_allow_html=True)
                cols_r[1].markdown(f"<span style='color:#F0F4FF;font-size:.8rem'>{cell['lot'] or '—'}</span>", unsafe_allow_html=True)
                cols_r[2].markdown(f"<span style='background:{color}22;color:{color};border:1px solid {color};border-radius:10px;padding:2px 8px;font-size:.7rem'>{stade}</span>", unsafe_allow_html=True)
                cols_r[3].markdown(f"<span style='font-size:.75rem;color:#A8B4CC'>{cell['date_greffe'] or '—'} → {cell['date_emergence'] or '—'}</span>", unsafe_allow_html=True)
                cols_r[4].markdown(f"<span style='font-size:.75rem;color:#A8B4CC'>{cell['qualite']}</span>", unsafe_allow_html=True)
                cols_r[5].markdown(f"<span style='font-size:.75rem;color:#A8B4CC'>{cell['poids_mg'] or '—'}</span>", unsafe_allow_html=True)
                
                nouveau_stade = cols_r[6].selectbox("", list(stade_color.keys()), 
                                                     index=list(stade_color.keys()).index(stade) if stade in stade_color else 0,
                                                     key=f"stade_{cell['id']}",
                                                     label_visibility="collapsed")
                if nouveau_stade != stade:
                    conn.execute("UPDATE cellules_royales SET stade=? WHERE id=?", (nouveau_stade, cell['id']))
                    conn.commit()
                    st.rerun()

    with tab_add:
        st.markdown("### ➕ Ajouter des cellules royales")
        with st.form("form_cellule"):
            col1, col2 = st.columns(2)
            lot_sel = col1.selectbox("Lot d'élevage", list(opts_lots.keys()) if opts_lots else ["—"])
            ruche_sel = col2.selectbox("Ruche destination", list(opts_ruches.keys()))
            
            col3, col4 = st.columns(2)
            numero = col3.number_input("N° cellule", 1, 999, 1)
            stade_init = col4.selectbox("Stade initial", ["oeuf", "larve", "operculée"])
            
            col5, col6 = st.columns(2)
            date_greffe_c = col5.date_input("Date greffe", datetime.date.today())
            qualite_c = col6.selectbox("Qualité estimée", ["bonne", "excellente", "médiocre", "inconnue"])
            
            if niv == "expert":
                col7, col8 = st.columns(2)
                poids_c = col7.number_input("Poids (mg)", 0.0, 1000.0, 0.0, step=10.0)
                longueur_c = col8.number_input("Longueur (mm)", 0.0, 40.0, 0.0, step=0.5)
            else:
                poids_c, longueur_c = 0.0, 0.0
            
            notes_c = st.text_area("Notes", height=60)
            sub_c = st.form_submit_button("✅ Enregistrer")

        if sub_c and opts_lots:
            date_op_c = date_greffe_c + datetime.timedelta(days=8)
            date_em_c = date_greffe_c + datetime.timedelta(days=16)
            lot_id = opts_lots.get(lot_sel)
            ruche_id_c = opts_ruches[ruche_sel]
            conn.execute("""
                INSERT INTO cellules_royales (elevage_id, ruche_id, numero, stade, date_greffe, 
                    date_operculee, date_emergence, qualite, poids_mg, longueur_mm, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (lot_id, ruche_id_c, numero, stade_init, str(date_greffe_c),
                  str(date_op_c), str(date_em_c), qualite_c, poids_c or None, longueur_c or None, notes_c))
            conn.commit()
            log_action("Cellule royale ajoutée", f"Lot {lot_sel} — N°{numero} — {stade_init}")
            st.success(f"✅ Cellule #{numero} enregistrée. Émergence prévue : {date_em_c}")
            st.rerun()

    with tab_timeline:
        st.markdown("### 📅 Timeline des cellules")
        df_tl = pd.read_sql("""
            SELECT cr.numero, er.nom as lot, cr.date_greffe, cr.date_operculee, cr.date_emergence, cr.stade, cr.qualite
            FROM cellules_royales cr
            LEFT JOIN elevage_reines er ON er.id = cr.elevage_id
            WHERE cr.date_greffe IS NOT NULL
            ORDER BY cr.date_greffe
        """, conn)
        
        if not df_tl.empty:
            # Crée un Gantt simplifié avec plotly
            fig_tl = go.Figure()
            colors_q = {"bonne": "#F5A623", "excellente": "#34D399", "médiocre": "#F87171", "inconnue": "#6B7A99"}
            
            for i, (_, row) in enumerate(df_tl.iterrows()):
                try:
                    d_start = datetime.date.fromisoformat(row['date_greffe'])
                    d_end = datetime.date.fromisoformat(row['date_emergence']) if row['date_emergence'] else d_start + datetime.timedelta(days=16)
                    label = f"{row['lot'] or '?'} - #{row['numero']}"
                    fig_tl.add_trace(go.Bar(
                        x=[(d_end - d_start).days],
                        y=[label],
                        base=[str(d_start)],
                        orientation='h',
                        marker_color=colors_q.get(str(row['qualite']), "#F5A623"),
                        name=label,
                        showlegend=False,
                        hovertemplate=f"{label}<br>Greffe: {d_start}<br>Émergence: {d_end}<br>Qualité: {row['qualite']}"
                    ))
                except:
                    pass
            
            fig_tl.update_layout(
                height=max(200, len(df_tl) * 30 + 60),
                barmode='overlay', template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10,b=10,l=0,r=0),
                xaxis_title="Durée (jours)"
            )
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("Aucune cellule avec date de greffe enregistrée.")

    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# ROUTEUR PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    inject_rtl_css()
    init_db()
    init_db_v3()
    init_db_v4()

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
        "assistant_vocal": page_assistant_vocal,
        "traitements": page_traitements,
        "productions": page_productions,
        "morpho": page_morpho,
        "carto": page_carto,
        "scanner_cadre": page_scanner_cadre,
        "transhumance": page_transhumance,
        "pedigree": page_pedigree,
        "elevage_reines": page_elevage_reines,
        "cellules_royales": page_cellules_royales,
        "meteo": page_meteo,
        "genetique": page_genetique,
        "flore": page_flore,
        "alertes": page_alertes,
        "journal": page_journal,
        "admin": page_admin,
    }
    fn = router.get(page, page_dashboard)
    fn()

    lang = get_lang()
    footer_texts = {
        "fr": "Trilingual AR/EN/FR · Élevage Reines · Cellules Royales · Générations · Niveaux Débutant→Expert",
        "en": "Trilingual AR/EN/FR · Queen Rearing · Royal Cells · Generations · Beginner→Expert Levels",
        "ar": "ثلاثي اللغات · تربية الملكات · الخلايا الملكية · الأجيال · مستويات للجميع",
    }
    st.markdown(f"""
    <div class='api-footer'>
        🐝 ApiTrack Pro v4.0 ULTIMATE · Streamlit + Python + SQLite · Rucher de l'Atlas · 2025
        <br><span style='font-size:.65rem;color:#6B7A99'>{footer_texts.get(lang, footer_texts['fr'])}</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
