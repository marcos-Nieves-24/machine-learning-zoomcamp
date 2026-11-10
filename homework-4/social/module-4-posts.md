# Module 4: Evaluation Metrics for Classification — posts

Built from the syllabus of the 2026 cohort (04-evaluation) and from the
numbers actually computed in `homework-4/homework-4.ipynb`. Nothing here is
aspirational: every figure appears in the executed notebook.

Verified facts used below:

- Dataset: `course_lead_scoring_2026.csv`, target `converted`.
- Split: 60/20/20 via `test_size=0.2` then `test_size=0.25`, both
  `random_state=1`.
- Q1 single-variable AUCs: `annual_income` 0.608,
  `number_of_courses_viewed` 0.723, `interaction_count` 0.765,
  `lead_score` 0.788. (`converted` against itself scores 1.000 — excluded,
  it answers nothing.)
- Q2 model: `LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)`,
  val AUC **0.732** (val accuracy 0.65).
- Q3 precision/recall intersection (min |P−R|, ignoring P=R=0): **0.63**
  (P≈0.733, R≈0.725).
- Q4 max F1 ≈ 0.758 at threshold **0.41**.
- Q5 5-fold AUCs (liblinear): [0.731, 0.730, 0.736, 0.741, 0.721],
  std **0.007**. With default `lbfgs`: std 0.013 + ConvergenceWarning.
- Q6: C=1e-06 → 0.617 ± 0.019; C=0.001 → **0.749 ± 0.010** (winner);
  C=1 → 0.732 ± 0.007.

---

## DEV (long-form, English)

Title: **Accuracy Lied to Me: Precision, Recall, ROC AUC and Cross-Validation in ML Zoomcamp Module 4**

In Module 3 I trained a lead-scoring classifier and reported 0.645 accuracy.
It sounded fine. Module 4 of ML Zoomcamp exists to destroy exactly that
feeling — accuracy hides everything interesting when classes are imbalanced.
Here is what I actually learned, with the numbers from my homework notebook.

### 1. AUC as a feature-importance flashlight

Before training anything, I scored every numerical column by using it raw as
the prediction and computing ROC AUC against the target (negating it when AUC
< 0.5, which just means negative correlation):

- `annual_income`: 0.608
- `number_of_courses_viewed`: 0.723
- `interaction_count`: 0.765
- `lead_score`: 0.788 ← winner

Two lessons. First, `lead_score` alone already separates the classes decently
— good to know before blaming the model. Second, the trap: if you loop over
all numeric columns naively you also score `converted` against itself and get
a perfect 1.000. I had that bug. Exclude the target.

### 2. The model: AUC 0.732, and why accuracy is not the story

One-hot encoding with `DictVectorizer` (fit on train only), logistic
regression with `solver="liblinear", C=1.0`, probabilities from
`predict_proba`. Validation AUC: **0.732**.

AUC'sasts interpretation is my favourite thing in the module: it is the
probability that a random positive outranks a random negative. My model ranks
a random converter above a random non-converter 73% of the time. One number,
no threshold chosen, immune to class balance. Accuracy can't say any of that.

### 3. The threshold tradeoff is real (plot it!)

Sweeping thresholds 0.00–1.00 in 0.01 steps: precision climbs, recall falls,
they cross at **0.63** (P≈0.733, R≈0.725). F1 peaks earlier, at **0.41**
(F1≈0.758). The default 0.5 threshold is optimal for nothing — it is just the
default. If missing a converter costs you more than a wasted email, you live
left of 0.5; if spam costs more, right of it. The plot decides, not the
default.

### 4. Cross-validation: std is the point, and the solver is the trap

5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=1)`), fresh
vectorizer per fold, AUC per fold: 0.731, 0.730, 0.736, 0.741, 0.721.
Mean ≈ 0.732, std **0.007**. The std is the deliverable — it says how much
I'd trust this number on new data.

And here is the bug that would have failed the homework: the spec says
`solver="liblinear"`. Writing just `LogisticRegression(max_iter=1000)` silently
uses `lbfgs`, which throws `ConvergenceWarning` on this data and lands on std
**0.013** — a different multiple-choice option. Same data, same seed, wrong
solver, wrong answer. Read the spec, copy the constructor exactly.

### 5. Tuning C: regularization you can see

Same 5-fold loop over C in [1e-06, 0.001, 1]:

- C=1e-06 → 0.617 ± 0.019 (regularization crushes the weights: underfit)
- C=0.001 → 0.749 ± 0.010 ← best
- C=1 → 0.732 ± 0.007

Tiny C is not "safer", it is a model that barely learns. The sweep makes the
bias-variance tradeoff visible instead of theoretical.

**Takeaway:** 0.65 accuracy told me my Module 3 model "works". Module 4 told
me what it actually does: ranks decently (0.732), forces a precision/recall
choice nobody made at 0.5, is stable across folds (±0.007), and tunes to
C=0.001. I trust the second description. Don't report accuracy alone again.

Repo: `<LINK>` · #mlzoomcamp

---

## LinkedIn (graphics-first carousel + caption)

Format: 5-card carousel (1080x1080). Cards 2 and 3 reuse the two plots from
the executed notebook (`Metrics vs Threshold`, `F1 score vs Threshold`).

**Card 1 — title:**
"Accuracy said my model works. These 4 plots disagreed."
Module 4 · ML Zoomcamp · lead scoring, n=5000

**Card 2 — graphic: Metrics vs Threshold plot.**
"Precision ↑, recall ↓. They cross at 0.63 (P≈0.73, R≈0.73).
The default 0.5 threshold is optimal for nothing."

**Card 3 — graphic: F1 vs Threshold plot.**
"F1 peaks at 0.41 (0.758) — far from 0.5.
Your threshold is a business decision, not a default."

**Card 4 — text card, the CV table:**
"Same model, 5 folds: 0.731 · 0.730 · 0.736 · 0.741 · 0.721 → std 0.007.
Tuning C: 1e-06 → 0.617 · 0.001 → 0.749 ✅ · 1 → 0.732.
Tiny C doesn't mean safer — it means underfit."

**Card 5 — text card, the gotcha:**
"The bug that fails the homework: `LogisticRegression(max_iter=1000)`
defaults to lbfgs → ConvergenceWarning → std 0.013 (wrong option).
The spec says solver='liblinear' → std 0.007 (right option).
Copy the constructor exactly. 🤏"

**Caption:**

I just completed Module 4: Evaluation Metrics for Classification from ML
Zoomcamp 2026 🎉 (carousel with my actual plots 👆)

This module covered:

🔹 Precision vs recall and why they fight each other

🔹 ROC curves and what AUC really means (P(random + outranks random −))

🔹 F1 and threshold selection as a business decision

🔹 K-fold cross-validation: the std is the deliverable

🔹 Tuning C and watching regularization underfit live

For the homework I evaluated my Module 3 lead-scoring model: val AUC 0.732,
precision/recall crossing at 0.63, F1 peaking at 0.41, 5-fold std 0.007, best
C=0.001 at 0.749 ± 0.010.

✨ My key takeaway: my Module 3 post celebrated 0.645 accuracy. This module
showed me that number hid every decision that matters — which errors I make,
at which threshold, and how stable the score is. And the pettiest lesson:
one wrong solver default (`lbfgs` vs `liblinear`) flips the homework answer
from 0.007 to 0.013. Specs are load-bearing.

➡️ Next step → Module 5. The model is evaluated; now I want it to be a good
model, not just a measured one.

#mlzoomcamp @Alexey Grigorev

(Repo link in the first comment — links in the post body kill reach.)

---

## Instagram (English, short + visual)

Single image or carousel with the two notebook plots. Caption:

Module 4 done: evaluation metrics 🎯

That 0.645 accuracy from last module was hiding everything: precision and
recall cross at 0.63, F1 peaks at 0.41 (far from the 0.5 default 👀), and
cross-validation says my model is stable (std 0.007 across 5 folds).

Best C: 0.001 → AUC 0.749. And the bug that almost cost me the homework:
skip `solver='liblinear'` and sklearn falls back to `lbfgs`, throws a
ConvergenceWarning, and hands you a different answer. Defaults need reading
too. 🤏

Swipe for the curves 👉

#mlzoomcamp #machinelearning #datascience #python

---

## X (single tweet, follows the homework template)

```text
Evaluated a classifier in Module 4 of #mlzoomcamp.

✅ Compared ROC AUC (0.732) and threshold metrics
✅ P/R cross at 0.63, F1 peaks at 0.41
✅ 5-fold CV tuned C=0.001 (0.749)

My repo: <LINK>
```

Fits in 280 characters. Optional follow-up reply with the gotcha: "PS: the
spec says solver='liblinear' — default lbfgs gives a different std (0.013 vs
0.007). Copy the constructor exactly."

---

## LinkedIn — Q6 code-snippet post (pair with the snippet image)

Suggested image: the Q6 cell — the `for C in [0.000001, 0.001, 1]` loop with
the per-fold `DictVectorizer` refit and `LogisticRegression(solver='liblinear',
C=C, max_iter=1000)`. Dark theme, font ≥ 18pt, cropped to the loop only.

```text
Most "tune C" snippets leak. This one doesn't. 👇

Question 6 of ML Zoomcamp Homework 4: 5-fold cross-validation over
C in [0.000001, 0.001, 1], same KFold(n_splits=5, shuffle=True,
random_state=1) every time.

Three things this snippet gets right:

1️⃣ Fresh DictVectorizer per fold — fit on the train fold, transform on the
val fold. Fit once outside the loop and your "validation" scores are lying.

2️⃣ The solver is in the constructor — solver='liblinear', exactly as the
spec demands. The default (lbfgs) converges differently here and answers a
different question.

3️⃣ Mean AND std, both rounded — because tuning on mean alone is how you
pick a C that was lucky once.

Results from my run:
▫️ C=1e-06 → 0.617 ± 0.019 (regularization crushes it: underfit)
▫️ C=0.001 → 0.749 ± 0.010 ✅
▫️ C=1 → 0.732 ± 0.007

Winner: C=0.001, no tie-break needed (best mean outright — lowest std, then
smallest C, only break ties).

Full notebook + FAQ in the first comment. 🔗

#mlzoomcamp @Alexey Grigorev
```

---

## X — Q6 code-snippet thread (each tweet ≤ 280 chars)

**Tweet 1 — hook (attach the snippet image here):**

```text
Most "tune C" snippets leak. This one doesn't. 🧵

ML Zoomcamp HW4 Q6: 5-fold CV over C in [1e-06, 0.001, 1].
```

**Tweet 2 — what it gets right:**

```text
3 things it gets right:
1️⃣ Fresh DictVectorizer per fold (fit train / transform val)
2️⃣ solver='liblinear' in the constructor — default lbfgs answers a different question
3️⃣ Mean AND std — don't pick a C that got lucky once
```

**Tweet 3 — results + link:**

```text
Results:
▫️ C=1e-06 → 0.617 ± 0.019 (underfit)
▫️ C=0.001 → 0.749 ± 0.010 ✅
▫️ C=1 → 0.732 ± 0.007

Winner: C=0.001. Notebook + FAQ: <LINK> #mlzoomcamp
```

---

## Instagram — Q6 code-snippet post (English)

Visual: snippet image (slide 1 = the C loop, slide 2 = the 3-line results
table). Code must be readable at phone size — dark theme, large font.

```text
Your "tune C" snippet is probably leaking. 👀

ML Zoomcamp HW4 Q6: 5-fold CV over C = [1e-06, 0.001, 1].

✅ Fresh DictVectorizer per fold (fit train / transform val)
✅ solver='liblinear' — the default lbfgs gives a different answer
✅ Mean AND std, never mean alone

C = 0.001 wins: 0.749 ± 0.010 🏆

Full code in the slides 👆 repo in bio 🔗

#mlzoomcamp #machinelearning #datascience #python #coding
```

---

## LinkedIn — metrics plot single-image post (English)

Visual: the `Metrics vs Threshold` figure from the executed notebook
(precision climbing, recall falling, F1 hump, crossing at 0.63). Export it
large — text must survive the feed.

```text
3 curves. 0 defaults survived. 📉

Precision, recall and F1 across every threshold 0.00 → 1.00 — from my
ML Zoomcamp Homework 4 model (val AUC 0.732).

What you're seeing:
📈 Precision climbs — raise the bar, fewer false alarms
📉 Recall falls — raise the bar, more converters missed
🤝 They cross at 0.63 — the only "balanced" point, and it's not 0.5
🏔️ F1 peaks at 0.41 — the best compromise lives left of default

The 0.5 threshold is not a decision. It's the absence of one. Every point
on these curves is a business tradeoff: cost of a wasted email vs cost of
a lost customer.

Plot it. Don't guess it.

Notebook + FAQ in the first comment. 🔗

#mlzoomcamp @Alexey Grigorev
```
