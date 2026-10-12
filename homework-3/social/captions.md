# Captions listos para publicar — Homework 3 (ML Zoomcamp 2026)

Imágenes en `out/`. Regenerar todo con:

```bash
cd homework-3/social && python3 generate_social_cards.py
```

Cada caption incluye el **alt text** al final: pegalo en el campo de texto
alternativo de la plataforma (LinkedIn e Instagram lo piden al subir la imagen).

---

## LinkedIn — `linkedin-post-1200x1200.png`

Terminé el Homework 3 del ML Zoomcamp 2026: clasificación sobre un dataset de
lead scoring de 5,000 leads. Estas son las 6 respuestas, con los números que
respaldan cada una 👇

1. Moda de `industry` → **technology** (1,173 filas). Ojo: después de rellenar
   los faltantes, `NA` tiene 240 filas y NO es la moda.
2. Mayor correlación → **interaction_count × lead_score** (r = 0.92).
3. Mayor información mutua → **lead_source** (MI = 0.03).
4. Accuracy en validación → **0.65** (0.645 exacto, `liblinear`, C=1.0).
5. Feature menos útil → **number_of_courses_viewed**: sacarla sólo baja 0.002
   de accuracy. Sacar `interaction_count`, en cambio, cuesta 0.044.
6. Mejor `C` → **0.001** (accuracy 0.645; con C más chico el modelo se
   regulariza tanto que baja a 0.598).

La lección que me llevé: en la Q5 la diferencia entre 0.643 y 0.642 parece
ruido, pero es determinista — mismo split, mismo seed. Lo verifiqué con dos
formas distintas de hacer one-hot encoding y dio idéntico.

¿Cuántas sacaste bien?

> Alt text: tarjeta oscura titulada "Clasificación: las 6 respuestas" con las
> seis respuestas del homework 3 de ML Zoomcamp: technology, interaction_count
> y lead_score, lead_source, 0.65, number_of_courses_viewed y C=0.001.

---

## Instagram — `instagram-post-1080x1350.png`

Las 6 respuestas del Homework 3 de clasificación 🎯

Lead scoring · 5,000 leads · split 60/20/20
LogisticRegression (liblinear, C=1.0)

1️⃣ Moda de industry → technology
2️⃣ Mayor correlación → interaction_count × lead_score (0.92)
3️⃣ Mayor MI → lead_source (0.03)
4️⃣ Accuracy → 0.65
5️⃣ Feature menos útil → number_of_courses_viewed (Δ 0.002)
6️⃣ Mejor C → 0.001

Guardalo para cuando lo hagas 💾

> Alt text: tarjeta oscura con las seis respuestas del homework 3 de
> clasificación del ML Zoomcamp, en dorado sobre fondo negro.

---

## Instagram Stories — `instagram-story-1080x1920.png`

Texto corto para Story (la imagen ya lleva el hook abajo):

¿Cuántas sacaste bien? 👀 Contame en los comentarios.

> Alt text: tarjeta vertical con las seis respuestas del homework 3 de
> clasificación y la pregunta "¿cuántas sacaste bien?".

---

## X (Twitter) — `x-card-1600x900.png`

Homework 3 del ML Zoomcamp 2026, resuelto 👇

1. Moda de industry → technology
2. Mayor correlación → interaction_count × lead_score (0.92)
3. Mayor MI → lead_source (0.03)
4. Accuracy en validación → 0.65
5. Feature menos útil → number_of_courses_viewed
6. Mejor C → 0.001

Lo interesante: sacar number_of_courses_viewed sólo cambia 0.002 de accuracy.
Sacar interaction_count te cuesta 0.044.

> Alt text: tarjeta apaisada con las seis respuestas del homework 3 de
> clasificación del ML Zoomcamp.

---

## Notas de publicación

- **LinkedIn**: subir la imagen sin link en el post (los enlaces bajan el
  alcance); si querés, dejá el link del repo en el primer comentario.
- **Instagram**: la versión 4:5 (1080x1350) ocupa más alto de pantalla que la
  cuadrada. La Story es 9:16 y ya trae el CTA abajo.
- **X**: 1600x900 se recorta a 16:9 en el feed; el texto queda dentro de la
  zona segura.
- Los números salen del notebook ejecutado (`homework-3.ipynb`), no están
  escritos a mano. Si cambiás el dataset, corré de nuevo
  `generate_social_cards.py` y las cards se actualizan solas.

---

## LinkedIn — carrusel de código (4 tarjetas)

Tarjetas: `linkedin-code-01-preparar-los-datos.png` →
`linkedin-code-02-el-modelo.png` → `linkedin-code-03-feature-menos-util.png` →
`linkedin-code-04-regularizacion.png`.

Subilas como post con varias imágenes (en el orden de la lista) y usá este texto:

El modelo de clasificación del Homework 3, en 4 snippets 👇

1️⃣ Preparar los datos: categóricas → "NA", numéricas → 0.0, y split 60/20/20 con
semillas fijas para que sea reproducible.

2️⃣ El modelo: DictVectorizer para el one-hot y
LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42).
Accuracy en validación: 0.645.

3️⃣ Feature elimination: entrenás sin cada feature y te quedás con la diferencia
más chica. Sacar number_of_courses_viewed sólo cuesta 0.002; sacar
interaction_count cuesta 0.044. Esa diferencia te dice qué feature no aporta.

4️⃣ Regularización: el barrido de C. Con C=0.000001 el modelo frena tanto que el
accuracy cae a 0.598; con C=0.001 llega a 0.645.

Deslizá → y contame: ¿cuál de los 4 snippets te resultó más útil?

> Alt text: cuatro tarjetas oscuras con código Python resaltado que muestran la
> preparación de datos, el entrenamiento del modelo de regresión logística, la
> eliminación de features y el barrido del parámetro C del homework 3.

**Nota de honestidad**: los snippets están abreviados para entrar en la tarjeta
(nombres de variables más cortos, llamadas partidas en dos líneas). El código
completo y ejecutado está en `homework-3/homework-3.ipynb`. No publiques estos
snippets como si fueran el notebook literal: son una versión de lectura.
