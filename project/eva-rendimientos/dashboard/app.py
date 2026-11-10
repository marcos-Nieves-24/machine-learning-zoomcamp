"""Dashboard Streamlit para EVA-Rendimientos (T6).

Consume el modelo LightGBM directamente (Booster nativo) + history.csv.
No requiere la API corriendo.

Tabs:
  1. Predicción individual (cultivo + municipio + año + área)
  2. Histórico de la serie (línea + predicción superpuesta)
  3. Mapa nacional (coroplético por municipio, para un cultivo)
  4. Comparación (top municipios, boxplot por departamento)
"""
import json
import unicodedata
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
API_DIR = ROOT / "api"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="EVA-Rendimientos Colombia", layout="wide",
                   page_icon="🌾")


@st.cache_resource
def load_artifacts():
    model = lgb.Booster(model_file=str(API_DIR / "model.txt"))
    config = json.loads((API_DIR / "feature_config.json").read_text())
    history = pd.read_csv(API_DIR / "history.csv")
    cultivar_map = pd.read_csv(API_DIR / "cultivar_map.csv")
    lookup = pd.read_csv(API_DIR / "lookup.csv")
    return model, config, history, cultivar_map, lookup


model, config, history, cultivar_map, lookup = load_artifacts()

CULT = cultivar_map.set_index("cultivo_norm").to_dict("index")
MUNI_LIST: dict = {}
for _, row in lookup.iterrows():
    MUNI_LIST.setdefault(row["municipio_norm"], []).append(row.to_dict())
MUNI_BY_COD = lookup.set_index("cod_mun").to_dict("index")

history["po"] = history["periodo_tipo"].map({"A": 0, "B": 1, "anual": 0})
history = history.sort_values(["cod_mun", "cultivo_norm", "anio", "po"])
history["grupo_norm"] = history["cultivo_norm"].map(
    {k: v["grupo_norm"] for k, v in CULT.items()})
history["municipio_norm"] = history["cod_mun"].map(
    {k: v["municipio_norm"] for k, v in MUNI_BY_COD.items()})
history["cod_dep"] = history["cod_mun"].map(
    {k: v["cod_dep"] for k, v in MUNI_BY_COD.items()})
history["r_lag1"] = history.groupby(
    ["cod_mun", "cultivo_norm"], sort=False)["rendimiento_clipped"].shift(1)

train_hist = history[history["anio"] <= 2020]
GMEAN = float(config["global_mean"])
SMOOTH_M = 10.0


def norm_key(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().casefold()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def smoothed(col):
    stats = train_hist.groupby(col)["rendimiento_clipped"].agg(["mean", "count"])
    return ((stats["count"] * stats["mean"] + SMOOTH_M * GMEAN)
            / (stats["count"] + SMOOTH_M)).to_dict()


ENC = {c: smoothed(c) for c in
       ["cultivo_norm", "municipio_norm", "cod_dep", "grupo_norm"]}
GRUPO_LAG1_MED = train_hist.groupby("grupo_norm")["r_lag1"].median().to_dict()
SERIES = {(k1, k2): g[["anio", "po", "rendimiento_clipped"]].to_numpy()
          for (k1, k2), g in history.groupby(["cod_mun", "cultivo_norm"], sort=False)}


def predict_one(cultivo, municipio, anio, periodo, area, departamento=None):
    cn = norm_key(cultivo)
    mn = norm_key(municipio)
    if cn not in CULT:
        return None, f"cultivo no cubierto: {cultivo}"
    cands = MUNI_LIST.get(mn, [])
    if not cands:
        return None, f"municipio no encontrado: {municipio}"
    if len(cands) > 1:
        if departamento:
            dept_norm = norm_key(departamento)
            cands = [c for c in cands if norm_key(c["departamento"]) == dept_norm]
        if len(cands) != 1:
            return None, f"municipio ambiguo: {municipio} (pasar departamento)"
    muni_row = cands[0]
    grupo = CULT[cn]["grupo_norm"]
    ciclo = CULT[cn]["ciclo_norm"]
    cod_mun = int(muni_row["cod_mun"])
    cod_dep = int(muni_row["cod_dep"])
    po_req = {"A": 0, "B": 1, "anual": 0}[periodo]
    serie = SERIES.get((cod_mun, cn))
    hist_len = 0
    past = np.array([])
    if serie is not None:
        mask = (serie[:, 0] < anio) | ((serie[:, 0] == anio) & (serie[:, 1] < po_req))
        past = serie[mask][:, 2].astype(float)
        hist_len = int(len(past))
    med = GRUPO_LAG1_MED.get(grupo, np.nan)
    if np.isnan(med):
        med = GMEAN
    def lag(k):
        return float(past[-k]) if len(past) >= k else float(med)
    def roll(k):
        return float(past[-k:].mean()) if len(past) >= 1 else float(med)
    feats = {
        "area_sembrada_log": float(np.log1p(area)),
        "is_B": int(periodo == "B"),
        "is_anual": int(periodo == "anual"),
        "hist_len": hist_len,
        "r_lag1": lag(1), "r_lag2": lag(2),
        "r_roll3": roll(3), "r_roll5": roll(5),
        "cultivo_norm_te": ENC["cultivo_norm"].get(cn, GMEAN),
        "municipio_norm_te": ENC["municipio_norm"].get(mn, GMEAN),
        "cod_dep_te": ENC["cod_dep"].get(cod_dep, GMEAN),
        "grupo_norm_te": ENC["grupo_norm"].get(grupo, GMEAN),
    }
    safe_grupo = grupo.replace(",", "").replace(" ", "_")
    for col in config["feature_columns"]:
        if col.startswith("grupo__"):
            feats[col] = int(safe_grupo == col[len("grupo__"):])
        elif col.startswith("ciclo__"):
            feats[col] = int(ciclo == col[len("ciclo__"):])
    X = pd.DataFrame([{c: feats.get(c, 0) for c in config["feature_columns"]}])
    pred = float(model.predict(X.to_numpy())[0])
    lo = float(max(pred + config["residual_q05"], 0.0))
    hi = float(pred + config["residual_q95"])
    conf = "alta" if hist_len >= 5 else ("media" if hist_len >= 1 else "baja")
    avisos = []
    if hist_len == 0:
        avisos.append("serie sin historial previo: predicción de baja confianza")
    elif hist_len < 3:
        avisos.append(f"serie corta (hist_len={hist_len}): confianza media")
    return {
        "municipio": muni_row["municipio"],
        "departamento": muni_row["departamento"],
        "rendimiento_predicho_t_ha": round(pred, 2),
        "intervalo_90": [round(lo, 2), round(hi, 2)],
        "hist_len": hist_len,
        "confianza": conf,
        "avisos": avisos,
    }, None


def predict_batch(cultivo, anio, periodo, area, municipios):
    cn = norm_key(cultivo)
    if cn not in CULT:
        return None, f"cultivo no cubierto: {cultivo}"
    results = []
    for mun in municipios:
        r, err = predict_one(cultivo, mun, anio, periodo, area)
        if err:
            results.append({"municipio": mun, "error": err})
        else:
            results.append(r)
    return results, None


# --- Sidebar ---
st.sidebar.title("🌾 EVA-Rendimientos")
st.sidebar.markdown("Predicción de rendimiento agrícola (t/ha) por municipio y cultivo en Colombia.")

cultivos_display = sorted(cultivar_map["cultivo"].tolist())
cultivos_norm = cultivar_map["cultivo_norm"].tolist()
display_to_norm = dict(zip(cultivos_display, cultivos_norm))

# Sidebar usa nombres display
cultivo_sel = st.sidebar.selectbox("Cultivo", cultivos_display, index=0)
cultivo_norm_sel = display_to_norm[cultivo_sel]
municipios_all = sorted(lookup["municipio"].tolist())
municipio_sel = st.sidebar.selectbox("Municipio", municipios_all, index=municipios_all.index("Campoalegre") if "Campoalegre" in municipios_all else 0)
anio_sel = st.sidebar.slider("Año", 2024, 2030, 2026)
periodo_sel = st.sidebar.selectbox("Periodo", ["A", "B", "anual"], index=0)
area_sel = st.sidebar.number_input("Área sembrada (ha)", min_value=0.0, value=100.0, step=10.0)
departamento_sel = st.sidebar.text_input("Departamento (solo si municipio ambiguo)", value="")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predicción", "📈 Histórico", "🗺️ Mapa nacional", "📋 Comparación"])

# Tab 1: Predicción individual
with tab1:
    st.header("Predicción individual")
    if st.button("Predecir", type="primary"):
        with st.spinner("Calculando..."):
            res, err = predict_one(cultivo_sel, municipio_sel, anio_sel, periodo_sel,
                                  area_sel, departamento_sel or None)
        if err:
            st.error(err)
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Rendimiento predicho", f"{res['rendimiento_predicho_t_ha']} t/ha")
            col2.metric("Intervalo 90%", f"[{res['intervalo_90'][0]}, {res['intervalo_90'][1]}]")
            col3.metric("Confianza", res["confianza"].capitalize())
            if res["avisos"]:
                for a in res["avisos"]:
                    st.warning(a)
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["rendimiento_predicho_t_ha"],
                title={"text": f"Rendimiento {cultivo_sel} — {res['municipio']}"},
                gauge={"axis": {"range": [0, max(res["rendimiento_predicho_t_ha"] * 2, 10)]},
                       "bar": {"color": "darkgreen"},
                       "steps": [{"range": [0, res["rendimiento_predicho_t_ha"]], "color": "lightgreen"}]},
            ))
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Histórico de la serie
with tab2:
    st.header("Histórico de la serie")
    cn = norm_key(cultivo_sel)
    mn = norm_key(municipio_sel)
    cands = MUNI_LIST.get(mn, [])
    if len(cands) == 1 or (departamento_sel and len(cands) > 1):
        if len(cands) > 1 and departamento_sel:
            cands = [c for c in cands if norm_key(c["departamento"]) == norm_key(departamento_sel)]
        if cands:
            cod_mun = int(cands[0]["cod_mun"])
            serie = history[(history["cod_mun"] == cod_mun) & (history["cultivo_norm"] == cn)]
            if len(serie) > 0:
                serie = serie.sort_values(["anio", "po"])
                fig = px.line(serie, x="anio", y="rendimiento_clipped",
                              color="periodo_tipo", markers=True,
                              title=f"Rendimiento histórico: {cultivo_sel} en {cands[0]['municipio']}",
                              labels={"rendimiento_clipped": "Rendimiento (t/ha)",
                                      "anio": "Año"})
                # Superponer predicción
                res, _ = predict_one(cultivo_sel, municipio_sel, anio_sel, periodo_sel,
                                     area_sel, departamento_sel or None)
                if res:
                    fig.add_scatter(x=[anio_sel], y=[res["rendimiento_predicho_t_ha"]],
                                    mode="markers", marker=dict(size=15, color="red",
                                    symbol="star"), name="Predicción")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Serie con {len(serie)} observaciones. "
                         f"Media histórica: {serie['rendimiento_clipped'].mean():.2f} t/ha")
            else:
                st.info("No hay datos históricos para esta combinación.")
        else:
            st.warning("Municipio no encontrado o ambiguo.")
    else:
        st.warning("Seleccioná un municipio válido.")

# Tab 3: Mapa nacional
with tab3:
    st.header("Mapa nacional de rendimiento predicho")
    cultivo_mapa = st.selectbox("Cultivo para el mapa", cultivos_display, index=0, key="mapa_cultivo")
    anio_mapa = st.slider("Año", 2024, 2030, 2026, key="mapa_anio")
    area_mapa = st.number_input("Área sembrada (ha)", min_value=0.0, value=50.0, step=10.0, key="mapa_area")
    if st.button("Generar mapa", type="primary"):
        with st.spinner("Calculando predicciones para todos los municipios..."):
            cands_all = lookup["municipio"].tolist()
            res_list, _ = predict_batch(cultivo_mapa, anio_mapa, "A", area_mapa, cands_all)
            if res_list:
                df_map = pd.DataFrame([r for r in res_list if "error" not in r])
                if len(df_map) > 0:
                    # Merge con coordenadas (aproximadas por departamento para simplicidad)
                    fig = px.choropleth(
                        df_map, locations="departamento", color="rendimiento_predicho_t_ha",
                        hover_name="municipio", hover_data=["rendimiento_predicho_t_ha", "confianza"],
                        title=f"Rendimiento predicho: {cultivo_mapa} ({anio_mapa})",
                        color_continuous_scale="YlGn",
                        labels={"rendimiento_predicho_t_ha": "t/ha"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_map[["municipio", "departamento", "rendimiento_predicho_t_ha",
                                        "intervalo_90", "confianza"]].sort_values(
                        "rendimiento_predicho_t_ha", ascending=False).head(20),
                                 use_container_width=True)
                else:
                    st.error("No se pudieron calcular predicciones.")

# Tab 4: Comparación
with tab4:
    st.header("Comparación por departamento")
    cultivo_comp = st.selectbox("Cultivo", cultivos_display, key="comp_cultivo")
    anio_comp = st.slider("Año", 2024, 2030, 2026, key="comp_anio")
    area_comp = st.number_input("Área sembrada (ha)", min_value=0.0, value=50.0, step=10.0, key="comp_area")
    if st.button("Calcular comparación", type="primary"):
        with st.spinner("Calculando..."):
            cands_all = lookup["municipio"].tolist()
            res_list, _ = predict_batch(cultivo_comp, anio_comp, "A", area_comp, cands_all)
            if res_list:
                df_comp = pd.DataFrame([r for r in res_list if "error" not in r])
                if len(df_comp) > 0:
                    # Boxplot por departamento
                    fig_box = px.box(df_comp, x="departamento", y="rendimiento_predicho_t_ha",
                                     title=f"Distribución de rendimiento predicho: {cultivo_comp} ({anio_comp})",
                                     labels={"rendimiento_predicho_t_ha": "t/ha"})
                    fig_box.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)
                    # Top 10
                    top10 = df_comp.nlargest(10, "rendimiento_predicho_t_ha")
                    fig_bar = px.bar(top10, x="rendimiento_predicho_t_ha", y="municipio",
                                     color="departamento", orientation="h",
                                     title="Top 10 municipios por rendimiento predicho",
                                     labels={"rendimiento_predicho_t_ha": "t/ha"})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    # Media nacional
                    media = df_comp["rendimiento_predicho_t_ha"].mean()
                    st.metric("Media nacional predicha", f"{media:.2f} t/ha",
                              f"{len(df_comp)} municipios")
