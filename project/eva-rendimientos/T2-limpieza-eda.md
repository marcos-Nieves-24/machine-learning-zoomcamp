# T2 — Limpieza y EDA EVA (2026-11-09)

**Estado:** hecho. Limpieza reproducible con `src/clean.py`, EDA ejecutado en `notebooks/02_cleaning_eda.ipynb`.
**Salidas:** `data/eva_clean/eva_clean.csv`, `data/eva_clean/cleaning_log.json`.

## Decisiones aplicadas (evidencia T1)

1. **Normalización de texto** (`norm_key`: casefold + sin acentos): fusionó 388 cultivos crudos → 248 normalizados (MAIZ/Maíz). Ciclo en minúsculas: transitorio/permanente/anual.
2. **Periodo:** `periodo_tipo` (anual / A / B) + `periodo_num` para ordenar series semestrales.
3. **Consistencia técnica** (UPRA: producción = rendimiento × área cosechada): error relativo >20% → 3.631 filas (0,97%) marcadas `inconsistente`. Se conservan con flag (el propio equipo UPRA las corrige con el municipio; para modelar se excluyen en T3).
4. **Target modelable:** rendimiento no nulo + área cosechada >0 → 361.322 filas (96,9%). Las 11.478 no modelables son área cosechada 0 / target nulo (cultivo perdido → rendimiento indefinido).
5. **Winsorización:** cota en p99,5 = 113,3 t/ha (p99 = 85). Afecta 1.845 filas (0,5%). Ceros conservados como señal de pérdida… aunque en la práctica todos los ceros coincidieron con área cosechada 0 (rendimiento mínimo modelable = 0,02), así que los ceros quedan fuera del set modelable.
6. **Cultivo raro:** <100 filas → 119 cultivos, solo 3.148 filas (0,8%). Se filtran o agrupan por `grupo_norm` en T3.
7. **`ratio_cosecha`** (cosechada/sembrada): >1 en 2,75% (inconsistencia de reporte municipal documentada).

## EDA (notebook 02)

- Distribución: media 10,0, mediana 6,0 t/ha; cola pesada hasta 253 (winsorizada en 113,3).
- Cobertura: 2006 (4.072 filas) → 2025 (25.617); 1.109 municipios, 32 departamentos.
- Top cultivos: maíz, fríjol, yuca, tomate, plátano, papa, arroz, café, caña panelera, cacao.
- Mediana de rendimiento por grupo varía de 1,05 (otros permanentes) a niveles altos en oleaginosas/frutales → target encoding por grupo/cultivo justificado para T3.
- Ciclo transitorio vs permanente con perfiles distintos → incluir `ciclo_norm` como feature.

## Para T3

- Features temporales: lags (1, 2) y medias móviles (3, 5 años) sobre `rendimiento_clipped`, por serie municipio×cultivo (respetando periodo A/B).
- Excluir: filas `inconsistente`, cultivos raros (o agrupar por `grupo_norm`), filas no modelables.
- Validación temporal: train 2006–2020, validación 2021–2022, test 2023–2024; walk-forward en T4.
- Baselines T3: media histórica por cultivo/municipio + LinearRegression + RandomForest (RMSE, MAE, R², MAPE).
