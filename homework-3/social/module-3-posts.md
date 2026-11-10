# Module 3: Machine Learning for Classification — posts

Built from the syllabus of the 2026 cohort (03-classification) and from the
numbers actually computed in `homework-3/homework-3.ipynb`. Nothing here is
aspirational: every figure appears in the executed notebook.

Verified facts used below:

- Dataset: `course_lead_scoring_2026.csv`, 5,000 rows, 57.66% `converted`.
- Split: 4000 full train / 3000 train / 1000 val / 1000 test (60/20/20).
- Model: `LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)`.
- Validation accuracy: 0.645.
- Majority-class baseline on the validation split: 0.597 (always answer
  "converted"), so the model is 4.8 points above it.
- Biggest correlation: `interaction_count` x `lead_score` = 0.9157.
- Biggest mutual information score with the target: `lead_source` = 0.03.
- Feature elimination: dropping `number_of_courses_viewed` costs 0.002,
  dropping `lead_source` costs 0.003, dropping `interaction_count` costs 0.044.
- Best `C` in the sweep: 0.001 (accuracy 0.645).

---

## LinkedIn

I just completed Module 3: Machine Learning for Classification from ML Zoomcamp 2026 🎉

This module covered:

🔹 Training and using logistic regression models

🔹 Encoding categorical variables with one-hot encoding

🔹 Measuring feature importance with the mutual information score

🔹 Selecting useful features through feature elimination

🔹 Interpreting what a classification model actually learned

For the homework I built a lead-scoring classifier on 5,000 leads: one-hot encoding with DictVectorizer, LogisticRegression with the liblinear solver, and a 60/20/20 split with fixed seeds. Result: 0.645 accuracy on validation.

✨ My key takeaway: correlation and mutual information answer different questions. The biggest correlation in the dataset was interaction_count × lead_score (0.92), but the feature with the highest mutual information score with the target was lead_source — a categorical feature that a correlation matrix literally cannot see. If I had ranked features by correlation alone, I would have gotten it wrong.

🤔 Something I found interesting: in the feature elimination exercise, removing number_of_courses_viewed cost 0.002 accuracy, while removing interaction_count cost 0.044. A gap that small (0.643 vs 0.642) looked like noise to me, so I re-ran the whole comparison with a second one-hot encoding approach to confirm it was deterministic and not an artifact of the library. It was. Small differences deserve verification before you dismiss them as noise.

➡️ Next step → Module 4: Evaluation Metrics for Classification. Here is the honest reason: 0.645 accuracy sounds fine until you notice that always predicting "converted" scores 0.597 on this validation split. My model is about 5 points better than a biased coin. I want precision, recall and ROC curves before I trust another classifier.

#mlzoomcamp @Alexey Grigorev

---

## Twitter / X

### Single tweet

Just wrapped up Module 3 of #mlzoomcamp 🚀

Learned about:

🔹 Logistic regression and the sigmoid

🔹 One-hot encoding categorical features

🔹 Mutual information score for feature importance

🔹 Feature selection by elimination

🔹 What C actually controls in regularization

✨ Key takeaway: my biggest correlation was interaction_count × lead_score (0.92), but the feature with the highest mutual information with the target was lead_source — a categorical feature correlation can't see.

➡️ Next up: Module 4 (Evaluation)

@Al_Grigor

### Thread

**Starter**

Finished Module 3 of #mlzoomcamp: Classification ✅

Built a lead-scoring classifier on 5,000 leads: logistic regression + one-hot encoding, 0.645 accuracy on validation.

Here's what stuck with me 👇

**💡 Takeaway**

Correlation and mutual information are not the same question.

Biggest correlation in the data: interaction_count × lead_score (0.92)
Biggest mutual information with the target: lead_source (0.03)

lead_source is categorical. A correlation matrix cannot see it at all. If I had ranked features with correlation alone, I would have ranked them wrong.

**🤔 What I got wrong at first**

In the feature elimination exercise, removing number_of_courses_viewed cost 0.002 accuracy. Removing interaction_count cost 0.044.

That first number looked like noise, so I re-ran the comparison with a second one-hot encoding approach to make sure it wasn't a library artifact. It wasn't. Verify a small difference before calling it noise.

**🚀 Goal for Module 4**

Stop trusting accuracy alone.

0.645 sounds fine until you notice that always predicting "converted" scores 0.597 on this split. My model is roughly 5 points better than a biased coin.

Precision, recall and ROC curves next.

---

## Instagram

Módulo 3 terminado: Clasificación 🎯

Lead scoring sobre 5,000 leads · one-hot con DictVectorizer · LogisticRegression (liblinear) · accuracy 0.645 en validación

Lo que me llevé: la correlación más grande fue interaction_count × lead_score (0.92), pero el feature con mayor información mutua contra el target fue lead_source (0.03) — categórico, invisible para una matriz de correlación.

Lo siguiente: Módulo 4, métricas de evaluación. 0.645 suena bien hasta que te das cuenta que predecir siempre "converted" te da 0.597 en esta partición.

#mlzoomcamp #machinelearning

---

## Reel / Short / TikTok — video (10s)

File: `homework-3/social/out/hw3-answers-1080x1920.mp4`
(1080x1920, 9:16, 30 fps, 10.0 s, H.264 + silent AAC, ~1.4 MB)

Timeline: title 2.2s, six answers 1.5s each, end card 1.6s, seven 0.4s dips to
black. Every card carries a progress bar because feed viewers watch muted.

Regenerate with:

```bash
cd homework-3/social && python3 generate_video.py
```

Caption:

```text
Las 6 respuestas del Homework 3 de #mlzoomcamp en 10 segundos 🎯

Lead scoring sobre 5,000 leads · LogisticRegression (liblinear) · accuracy 0.645

¿Cuántas sacaste bien? Baja en los comentarios 👇

#mlzoomcamp #machinelearning
```

## TikTok — caption (English)

### Video version (`hw3-answers-1080x1920.mp4`)

```text
I trained a lead-scoring model for ML Zoomcamp Homework 3 and the accuracy
question humbled me.

0.645 accuracy sounds fine.
Always answering "converted" scores 0.597.
Five points. That is all my model bought.

The 6 answers 👇
1️⃣ technology
2️⃣ interaction_count × lead_score (r = 0.92)
3️⃣ lead_source (MI = 0.03)
4️⃣ 0.65
5️⃣ number_of_courses_viewed
6️⃣ C = 0.001

Favourite part: the biggest correlation was 0.92, but the most useful feature
was a categorical one the correlation matrix literally cannot see.

Sound off, it reads itself 🔇
How many did you get? 👇

#mlzoomcamp #machinelearning #datascience #python #learnontiktok
```

### Photo carousel version

TikTok photo mode: the first frame is the title card, so the caption can be
shorter.

```text
ML Zoomcamp Homework 3, all 6 answers ✅

1️⃣ technology
2️⃣ interaction_count × lead_score
3️⃣ lead_source
4️⃣ 0.65
5️⃣ number_of_courses_viewed
6️⃣ C = 0.001

swipe → the accuracy one is humbling: 0.645 against 0.597 for always
answering "converted"

save it for when you get there 🔖

#mlzoomcamp #machinelearning #datascience
```

### Hooks to A/B test (the first line decides the post)

1. "My model got 0.65 accuracy. Always guessing 'yes' gets 0.60."
2. "6 answers, 10 seconds, one very humbling accuracy."
3. "I almost deleted the right answer because it looked like noise."
4. "The feature that mattered most was invisible to the correlation matrix."

The clip is silent on purpose. TikTok autoplays muted anyway, and the card
carries every number on screen, so it reads without sound. Add a voiceover
later if reach needs it.

- **LinkedIn**: post without a link (links reduce reach); drop the repo link in
  the first comment instead. Pair with `linkedin-code-02-el-modelo.png`.
- **X**: the single tweet fits as-is. For the thread, attach
  `linkedin-code-03-feature-menos-util.png` to the 🤔 tweet.
- The 0.597 baseline is the honest edge of this model. Keep it in the copy: a
  post that says "0.65 accuracy, done" hides the part that actually taught you
  something.
- Numbers come from the executed notebook. If the notebook changes, update this
  file and the cards together.
