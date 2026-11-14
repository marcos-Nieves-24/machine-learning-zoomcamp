"""Streamlit dashboard for BeeBuzz — light theme, bilingual, user-oriented."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURES
from predict import predict_one, predict_type

# ---------- Translations ----------
T = {
    "en": {
        "title": "BeeBuzz",
        "subtitle": "Bee classifier by sound — upload your data and get predictions",
        "tab_predict": "Predict",
        "tab_results": "My results",
        "tab_about": "About model",
        "predict_header": "Make predictions",
        "input_mode": "Input mode",
        "upload_csv": "Upload CSV",
        "manual": "Manual input",
        "upload_label": "Upload a CSV with acoustic features",
        "rows_loaded": "rows loaded with",
        "columns": "columns",
        "predict_all": "Predict all",
        "predictions_done": "predictions completed",
        "download_results": "Download results (CSV)",
        "download_example": "Download example CSV",
        "load_example": "Load example data",
        "manual_desc": "Enter acoustic feature values:",
        "freq_duration": "Frequency and duration",
        "fundamental_freq": "Fundamental frequency (Hz)",
        "duration": "Duration (s)",
        "mfcc_mean": "MFCC — mean (1-13)",
        "mfcc_sd": "MFCC — std dev (1-13)",
        "mfcc_skew": "MFCC — skewness (1-13)",
        "mfcc_kurt": "MFCC — kurtosis (1-13)",
        "mfcc_med": "MFCC — median (1-13)",
        "predict_species": "Predict species",
        "predict_buzz": "Predict buzz type",
        "species_result": "Species",
        "confidence": "Confidence",
        "buzz_type_result": "Buzz type",
        "results_header": "My results",
        "no_predictions": "No predictions yet. Go to 'Predict' tab.",
        "no_predictions_hint": "Upload a CSV or enter values manually, then click Predict.",
        "total_predictions": "Total predictions",
        "most_frequent": "Most frequent species",
        "avg_confidence": "Average confidence",
        "species_dist": "Predicted species",
        "buzz_dist": "Buzz type",
        "conf_by_species": "Confidence by species",
        "results_table": "Results table",
        "clear_results": "Clear results",
        "about_header": "About the model",
        "about_text": """**BeeBuzz** classifies bee species from their buzzing sounds
(floral sonication and flight). It uses acoustic features (fundamental frequency,
MFCC) extracted from audio.

**Recognized species:**
- Banguela sp. (*Cadeguala occidentalis*)
- *Bombus morio*
- Centris cinza (*Centris cineraria*)
- Oxaea verdinha (*Oxaea flavescens*)
- Xylocopa pretinha (*Xylocopa* sp.)

**Performance (5-fold CV):**
- Species: 83.5% accuracy (RF, 67 features)
- Buzz type (floral/flight): 99.0% accuracy

**Data source:** Mendeley Data — Dual-Signal Buzz Pollination (CC BY 4.0).
""",
        "footer": "BeeBuzz — ML Zoomcamp 2026 Midterm | Dataset: Mendeley (CC BY 4.0)",
        "lang_label": "Language",
        "welcome_title": "Welcome to BeeBuzz",
        "welcome_text": "Classify bee species from acoustic features. Follow these steps:",
        "step1": "1. Upload a CSV or enter values manually",
        "step2": "2. Click 'Predict' to classify",
        "step3": "3. View results and statistics in 'My results'",
        "csv_invalid": "CSV columns do not match expected features. Check the example CSV.",
        "csv_valid": "CSV validated successfully.",
        "example_loaded": "Example data loaded. You can modify the values.",
        "onboarding_done": "Got it, don't show again",
        "show_all_fields": "Show all 67 fields",
        "key_features": "Key features (most important)",
    },
    "es": {
        "title": "BeeBuzz",
        "subtitle": "Clasificador de abejas por sonido — sube tus datos y obtén predicciones",
        "tab_predict": "Predecir",
        "tab_results": "Mis resultados",
        "tab_about": "Acerca del modelo",
        "predict_header": "Hacer predicciones",
        "input_mode": "Modo de entrada",
        "upload_csv": "Subir CSV",
        "manual": "Ingreso manual",
        "upload_label": "Sube un CSV con features acústicas",
        "rows_loaded": "filas cargadas con",
        "columns": "columnas",
        "predict_all": "Predecir todas",
        "predictions_done": "predicciones completadas",
        "download_results": "Descargar resultados (CSV)",
        "download_example": "Descargar CSV de ejemplo",
        "load_example": "Cargar datos de ejemplo",
        "manual_desc": "Ingresa los valores de las features acústicas:",
        "freq_duration": "Frecuencia y duración",
        "fundamental_freq": "Frecuencia fundamental (Hz)",
        "duration": "Duración (s)",
        "mfcc_mean": "MFCC — media (1-13)",
        "mfcc_sd": "MFCC — desviación estándar (1-13)",
        "mfcc_skew": "MFCC — skewness (1-13)",
        "mfcc_kurt": "MFCC — kurtosis (1-13)",
        "mfcc_med": "MFCC — mediana (1-13)",
        "predict_species": "Predecir especie",
        "predict_buzz": "Predecir tipo de buzz",
        "species_result": "Especie",
        "confidence": "Confianza",
        "buzz_type_result": "Tipo de buzz",
        "results_header": "Mis resultados",
        "no_predictions": "Aún no has hecho predicciones.",
        "no_predictions_hint": "Sube un CSV o ingresa valores manualmente, luego haz clic en Predecir.",
        "total_predictions": "Total de predicciones",
        "most_frequent": "Especie más frecuente",
        "avg_confidence": "Confianza promedio",
        "species_dist": "Especies predichas",
        "buzz_dist": "Tipo de buzz",
        "conf_by_species": "Confianza por especie",
        "results_table": "Tabla de resultados",
        "clear_results": "Limpiar resultados",
        "about_header": "Acerca del modelo",
        "about_text": """**BeeBuzz** clasifica especies de abejas a partir de sus sonidos de zumbido
(floral sonication y vuelo). Utiliza features acústicas (frecuencia fundamental,
MFCC) extraídas del audio.

**Especies reconocidas:**
- Banguela sp. (*Cadeguala occidentalis*)
- *Bombus morio*
- Centris cinza (*Centris cineraria*)
- Oxaea verdinha (*Oxaea flavescens*)
- Xylocopa pretinha (*Xylocopa* sp.)

**Rendimiento (5-fold CV):**
- Especies: 83.5% accuracy (RF, 67 features)
- Tipo de buzz (floral/flight): 99.0% accuracy

**Fuente de datos:** Mendeley Data — Dual-Signal Buzz Pollination (CC BY 4.0).
""",
        "footer": "BeeBuzz — ML Zoomcamp 2026 Midterm | Dataset: Mendeley (CC BY 4.0)",
        "lang_label": "Idioma",
        "welcome_title": "Bienvenido a BeeBuzz",
        "welcome_text": "Clasifica especies de abejas a partir de features acústicas. Sigue estos pasos:",
        "step1": "1. Sube un CSV o ingresa valores manualmente",
        "step2": "2. Haz clic en Predecir para clasificar",
        "step3": "3. Ve tus resultados y estadísticas en 'Mis resultados'",
        "csv_invalid": "Las columnas del CSV no coinciden con las features esperadas. Revisa el CSV de ejemplo.",
        "csv_valid": "CSV validado correctamente.",
        "example_loaded": "Datos de ejemplo cargados. Puedes modificar los valores.",
        "onboarding_done": "Entendido, no mostrar de nuevo",
        "show_all_fields": "Mostrar los 67 campos",
        "key_features": "Features clave (más importantes)",
    },
}

# ---------- Custom CSS (light theme) ----------
st.markdown("""
<style>
  .stApp { background-color: #f5f5f7; }
  .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }

  .bb-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 24px; margin-bottom: 20px;
    background: #ffffff; border: 1px solid #d8d8e0; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .bb-header h1 { margin: 0; color: #d4880a; font-size: 26px; font-weight: 700; line-height: 1.2; }
  .bb-header p { margin: 2px 0 0; color: #555; font-size: 13px; }

  .metric-card {
    background: #ffffff; border: 1px solid #d8d8e0; border-radius: 10px;
    padding: 14px 10px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .metric-card .val { font-size: 26px; font-weight: 700; color: #d4880a; }
  .metric-card .lbl { font-size: 11px; color: #777; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }

  .result-box {
    background: #ffffff; border: 1px solid #d4880a; border-radius: 10px;
    padding: 18px; text-align: center; margin: 10px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .result-box .species { font-size: 22px; font-weight: 700; color: #d4880a; }
  .result-box .conf { font-size: 13px; color: #555; margin-top: 4px; }

  .welcome-box {
    background: #fff8e7; border: 1px solid #f5a623; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 16px;
  }
  .welcome-box h3 { color: #d4880a; margin: 0 0 8px 0; font-size: 16px; }
  .welcome-box p { color: #555; margin: 0 0 8px 0; font-size: 13px; }
  .welcome-box ol { color: #333; margin: 0; padding-left: 20px; font-size: 13px; }
  .welcome-box li { margin-bottom: 4px; }

  section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #d8d8e0; }
  section[data-testid="stSidebar"] .stNumberInput input {
    background-color: #f5f5f7 !important; border: 1px solid #c8c8d0 !important;
    color: #1a1a2e !important; border-radius: 6px !important;
  }
  section[data-testid="stSidebar"] label { color: #333 !important; }

  .stButton > button[kind="primary"] {
    background-color: #d4880a !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 6px 18px !important;
  }
  .stButton > button[kind="primary"]:hover { background-color: #b8720a !important; color: #ffffff !important; }
  .stButton > button:not([kind="primary"]) {
    background-color: #ffffff !important; color: #1a1a2e !important;
    border: 1px solid #c8c8d0 !important; border-radius: 8px !important;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background-color: #ffffff; border-radius: 8px 8px 0 0; color: #555;
    border: 1px solid #d8d8e0; padding: 8px 18px;
  }
  .stTabs [aria-selected="true"] {
    background-color: #d4880a !important; color: #ffffff !important;
    font-weight: 600; border-color: #d4880a;
  }

  .streamlit-expanderHeader {
    background-color: #ffffff !important; border-radius: 8px !important;
    color: #1a1a2e !important; border: 1px solid #d8d8e0 !important;
  }

  .stDataFrame { border: 1px solid #d8d8e0; border-radius: 8px; overflow: hidden; }

  .footer {
    text-align: center; color: #888; font-size: 11px;
    margin-top: 32px; padding: 12px; border-top: 1px solid #d8d8e0;
  }

  h1, h2, h3 { color: #1a1a2e; }
  .stMarkdown p { color: #333; }
  .stMarkdown li { color: #333; }
  .stRadio label { color: #333 !important; }
  .stFileUploader { border: 1px dashed #c8c8d0; border-radius: 8px; padding: 12px; background: #ffffff; }
  .stProgress > div > div { background-color: #d4880a !important; }
  .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- Logo ----------
logo_path = ROOT / "dashboard" / "logo-light.svg"
if not logo_path.exists():
    logo_path = ROOT / "dashboard" / "logo.svg"
if logo_path.exists():
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" width="48" height="48" style="border-radius:8px"/>'
else:
    logo_html = ""

# ---------- Language selector ----------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_choice = st.selectbox(
        "Language / Idioma",
        ["English", "Español"],
        index=0 if st.session_state.lang == "en" else 1,
    )
    st.session_state.lang = "en" if lang_choice == "English" else "es"

lang = st.session_state.lang
t = T[lang]

# ---------- Onboarding ----------
if "onboarding_done" not in st.session_state:
    st.session_state.onboarding_done = False

if not st.session_state.onboarding_done:
    st.markdown(f"""
    <div class="welcome-box">
      <h3>{t['welcome_title']}</h3>
      <p>{t['welcome_text']}</p>
      <ol>
        <li>{t['step1']}</li>
        <li>{t['step2']}</li>
        <li>{t['step3']}</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)
    if st.button(t["onboarding_done"]):
        st.session_state.onboarding_done = True
        st.rerun()

# ---------- Header ----------
st.markdown(f"""
<div class="bb-header">
  {logo_html}
  <div>
    <h1>{t['title']}</h1>
    <p>{t['subtitle']}</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Config ----------
cfg = json.loads((ROOT / "models" / "feature_config.json").read_text())
classes = cfg["classes"]

# ---------- Session state ----------
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# ---------- Example data ----------
example_df = None
example_path = ROOT / "data" / "raw" / "buzz_data.csv"
if example_path.exists():
    example_df = pd.read_csv(example_path, sep=";", decimal=",").head(3)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs([t["tab_predict"], t["tab_results"], t["tab_about"]])

# ===================== TAB 1 =====================
with tab1:
    st.header(t["predict_header"])
    mode = st.radio(t["input_mode"], [t["upload_csv"], t["manual"]], horizontal=True)

    if mode == t["upload_csv"]:
        # Example CSV download
        if example_df is not None:
            example_csv = example_df.to_csv(index=False, sep=";", decimal=",")
            st.download_button(t["download_example"], example_csv, "example.csv", "text/csv")

        uploaded = st.file_uploader(t["upload_label"], type="csv")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, sep=";", decimal=",")
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded)

            # Validate columns
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error(f"{t['csv_invalid']} Missing: {', '.join(missing[:5])}...")
            else:
                st.success(t["csv_valid"])
                st.write(f"**{len(df)} {t['rows_loaded']} {len(df.columns)} {t['columns']}**")
                st.dataframe(df.head(5), use_container_width=True)

                if st.button(t["predict_all"], type="primary"):
                    results = []
                    progress = st.progress(0)
                    for i, (_, row) in enumerate(df.iterrows()):
                        feats = {f: float(row[f]) if f in row and pd.notna(row[f]) else 0.0 for f in FEATURES}
                        sp = predict_one(feats)
                        bt = predict_type(feats)
                        results.append({
                            "species": sp["species"],
                            "species_conf": round(sp["confidence"], 4),
                            "buzz_type": bt["buzz_type"],
                            "buzz_conf": round(bt["confidence"], 4),
                        })
                        progress.progress((i + 1) / len(df))
                    res_df = pd.DataFrame(results)
                    st.session_state.predictions.append(res_df)
                    st.success(f"{len(res_df)} {t['predictions_done']}")
                    st.dataframe(res_df, use_container_width=True)
                    csv_out = res_df.to_csv(index=False)
                    st.download_button(t["download_results"], csv_out, "predictions.csv", "text/csv")

    else:
        st.write(t["manual_desc"])

        # Load example button
        if example_df is not None:
            if st.button(t["load_example"]):
                row = example_df.iloc[0]
                for f in FEATURES:
                    st.session_state[f"val_{f}"] = float(row[f]) if f in row and pd.notna(row[f]) else 0.0
                st.success(t["example_loaded"])
                st.rerun()

        # Key features (most important) — always visible
        st.subheader(t["key_features"])
        vals = {}
        vals["ffreq"] = st.number_input(t["fundamental_freq"], value=st.session_state.get("val_ffreq", 239.0), format="%.2f", key="ffreq_input")
        vals["duration.s"] = st.number_input(t["duration"], value=st.session_state.get("val_duration.s", 0.3), format="%.4f", key="dur_input")

        # All fields toggle
        show_all = st.checkbox(t["show_all_fields"], value=False)
        if show_all:
            with st.expander(t["mfcc_mean"], expanded=False):
                for i in range(1, 14):
                    key = f"mean.mfc {i}"
                    vals[key] = st.number_input(key, value=st.session_state.get(f"val_{key}", 0.0), format="%.4f", key=f"inp_{key}")
            with st.expander(t["mfcc_sd"], expanded=False):
                for i in range(1, 14):
                    key = f"sd.mfc {i}"
                    vals[key] = st.number_input(key, value=st.session_state.get(f"val_{key}", 0.0), format="%.4f", key=f"inp_{key}")
            with st.expander(t["mfcc_skew"], expanded=False):
                for i in range(1, 14):
                    key = f"skew.mfc {i}"
                    vals[key] = st.number_input(key, value=st.session_state.get(f"val_{key}", 0.0), format="%.4f", key=f"inp_{key}")
            with st.expander(t["mfcc_kurt"], expanded=False):
                for i in range(1, 14):
                    key = f"kurt.mfc {i}"
                    vals[key] = st.number_input(key, value=st.session_state.get(f"val_{key}", 0.0), format="%.4f", key=f"inp_{key}")
            with st.expander(t["mfcc_med"], expanded=False):
                for i in range(1, 14):
                    key = f"med.mfc {i}"
                    vals[key] = st.number_input(key, value=st.session_state.get(f"val_{key}", 0.0), format="%.4f", key=f"inp_{key}")
        else:
            # Fill missing with 0
            for f in FEATURES:
                if f not in vals:
                    vals[f] = 0.0

        c1, c2 = st.columns(2)
        if c1.button(t["predict_species"], type="primary"):
            with st.spinner("Predicting..."):
                res = predict_one(vals)
            # Format species name: italicize genus/species, but not "sp."
            sp_name = res['species']
            sp_parts = sp_name.split()
            sp_formatted = " ".join(f"<i>{p}</i>" if p.lower() != "sp." else p for p in sp_parts)
            st.markdown(f"""
            <div class="result-box">
              <div class="species">{sp_formatted}</div>
              <div class="conf">{t['confidence']}: {res['confidence']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.bar_chart(pd.Series(res["probabilities"]))
            st.session_state.predictions.append(pd.DataFrame([{
                "species": res["species"],
                "species_confidence": res["confidence"],
                "buzz_type": "—",
                "buzz_confidence": 0.0,
            }]))

        if c2.button(t["predict_buzz"], type="primary"):
            with st.spinner("Predicting..."):
                res = predict_type(vals)
            st.markdown(f"""
            <div class="result-box">
              <div class="species">{res['buzz_type']}</div>
              <div class="conf">{t['confidence']}: {res['confidence']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.bar_chart(pd.Series(res["probabilities"]))

# ===================== TAB 2 =====================
with tab2:
    st.header(t["results_header"])
    if not st.session_state.predictions:
        st.info(t["no_predictions"])
        st.caption(t["no_predictions_hint"])
    else:
        all_preds = pd.concat(st.session_state.predictions, ignore_index=True)
        st.write(f"**{t['total_predictions']}: {len(all_preds)}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="val">{len(all_preds)}</div><div class="lbl">{t["total_predictions"]}</div></div>', unsafe_allow_html=True)
        with c2:
            top_sp = all_preds["species"].mode()[0] if len(all_preds) > 0 else "—"
            st.markdown(f'<div class="metric-card"><div class="val" style="font-size:16px">{top_sp}</div><div class="lbl">{t["most_frequent"]}</div></div>', unsafe_allow_html=True)
        with c3:
            avg_conf = all_preds["species_confidence"].mean()
            st.markdown(f'<div class="metric-card"><div class="val">{avg_conf:.1%}</div><div class="lbl">{t["avg_confidence"]}</div></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t["species_dist"])
            st.bar_chart(all_preds["species"].value_counts())
        with col2:
            st.subheader(t["buzz_dist"])
            st.bar_chart(all_preds["buzz_type"].value_counts())

        st.subheader(t["conf_by_species"])
        conf = all_preds.groupby("species")["species_confidence"].mean().sort_values(ascending=False)
        st.bar_chart(conf)

        st.subheader(t["results_table"])
        st.dataframe(all_preds, use_container_width=True)

        if st.button(t["clear_results"]):
            st.session_state.predictions = []
            st.rerun()

# ===================== TAB 3 =====================
with tab3:
    st.header(t["about_header"])
    st.markdown(t["about_text"])

    shap_summary = ROOT / "reports" / "shap_summary.png"
    shap_bar = ROOT / "reports" / "shap_bar.png"
    if shap_summary.exists():
        st.image(str(shap_summary), caption="SHAP — global feature importance")
    if shap_bar.exists():
        st.image(str(shap_bar), caption="SHAP — feature importance by species")
    if not shap_summary.exists():
        st.info("Run `python src/shap_analysis.py` to generate SHAP plots.")

# ---------- Footer ----------
st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
