# External probe: same-domain wavs through our pipeline (Nov 2026)

Question: what happens when real recordings from the *same* study go through
`scripts/extract_features.py` (librosa) and into the trained model?

Method: downloaded 5 raw `.wav` (one per species) from the companion GitHub repo
(Caarvelos/Bee-acoustics-classification, `data/all_audios/`), extracted the 67
features with our script, predicted with `models/model.joblib`.

| File (true species) | librosa ffreq | CSV ffreq | librosa mean.mfc1 | CSV mean.mfc1 | Predicted | Confidence |
|---|---|---|---|---|---|---|
| B_Ban_sp_1_1 (Banguela sp.) | 67 Hz | 239 Hz | -468 | -57 | Centris cinza | 0.51 |
| B_Bom_mor_1_1 (*Bombus morio*) | 52 Hz | 260 Hz | -471 | -55 | Banguela sp. | 0.33 |
| B_Cen_cin_1_1 (Centris cinza) | 54 Hz | 129 Hz | -459 | -55 | Centris cinza | 0.57 |
| B_Oxa_ver_1_1 (Oxaea verdinha) | 50 Hz | 323 Hz | -458 | -53 | Banguela sp. | 0.28 |
| B_Xyl_pre_1_1 (Xylocopa pretinha) | 59 Hz | 237 Hz | -412 | -46 | Banguela sp. | 0.29 |

Accuracy: **1/5 (20% = chance)**. Duration matches exactly (0.304 vs 0.304 s),
so I/O and resampling are correct. The gap is in feature extraction:

- librosa.yin locks onto ~50-67 Hz on these short noisy clips; the paper's
  pipeline reports 129-323 Hz (not a clean octave ratio, so not a simple
  octave error — likely different voicing detection).
- MFCC magnitudes differ by an order of magnitude (different filterbank scaling).

Conclusion: the `.wav → CSV → dashboard` plumbing works end to end, but
librosa features are **out of distribution** for this model. Fix (capstone):
re-extract features from the training `.wav`s with this same script and
retrain, or fit a per-feature standardization. Do not use audio-upload
predictions as field results until then.
