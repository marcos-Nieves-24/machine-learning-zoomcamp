# T1 — Ingesta y verificación EVA (2026-11-09)

**Estado:** hecho. Datos descargados desde datos.gov.co con `src/ingest.py` y verificados con `src/verify.py`.
**Reporte:** `data/verification_report.json`.

## Datos

| Bloque | ID Socrata | Filas | Cobertura |
|---|---|---|---|
| EVA 2006–2018 | `2pnw-mmge` | 206.068 | Histórico |
| EVA 2019–2025 | `uejq-wxrr` | 166.732 | Reciente (incluye 2025) |
| **Total** | | **372.800** | **2006–2025, 20 años** |

- Municipios: 1.109 | Departamentos: 32 | Cultivos: 388 | Grupos: 21
- Columnas unificadas: 19 (ver `src/ingest.py:COLUMN_MAP`; incluye `fuente` para trazabilidad de bloque)
- Archivos crudos: `data/eva_raw/eva_2006_2018.csv`, `data/eva_raw/eva_2019_2025.csv`, `data/eva_raw/manifest.json`

## Calidad verificada

- Nulos: `rendimiento_t_ha` 0,92% (3.433 filas), `nombre_cientifico` 2.857, `cod_cultivo` 206.068 (solo existe en el bloque reciente, esperado). Resto de columnas: 0 nulos.
- Target (`rendimiento_t_ha`): 369.367 válidos. Media 9,85, mediana 5,9 t/ha. p1 = 0, p99 = 86, máx 253.
- Ceros en target: 1,93% (7.116 filas, cultivos perdidos). Ceros en área cosechada: 3,08%.
- Consistencia: producción vs rendimiento × área cosechada cierra con error relativo máx 1,98 (hay inconsistencias puntuales para T2).
- Cultivos con <100 filas: 184 de 388 → T2 filtra o agrupa por grupo.
- Ciclo: TRANSITORIO/TRANSITORIO + PERMANENTE/PERMANENTE + ANUAL (mayúsculas en histórico, capitalizado en reciente → normalizar en T2).
- Cobertura anual creciente: 4.072 (2006) → 25.617 (2025).

## Decisiones para T2

1. Normalizar mayúsculas/minúsculas de `ciclo_cultivo` y nombres de cultivo (MAIZ vs Maíz duplican series).
2. Winsorizar rendimiento (q1–q99) o filtrar ceros extremos; tratar nulos de target (imputar vs excluir).
3. Filtrar cultivos con <100 filas o agrupar por `grupo_cultivo`.
4. Revisar filas con producción inconsistente (error relativo >20%) antes de modelar.
