# Cross-dataset probe (Nov 2026)

Question: what does the BeeBuzz model do with audio it was never trained on?
All probes run through the same pipeline: `extract_features.py` → model.

## Probe 1 — in-domain (same study, held-out clips)

5 raw `.wav` from the companion repo (one per species, not in training).
Result: **5/5 correct**, confidences 0.53–0.84.
The librosa retrain closed the pipeline gap (see EXTERNAL_PROBE.md).

## Probe 2 — cross-species, same family (Zenodo 10359686, honeybee audio)

4 × *Apis mellifera* 1-sec hive clips + 2 × drone clips (out-of-distribution
species, different recorder, different country).

| Clip | ffreq | Prediction | Confidence | Buzz type |
|---|---|---|---|---|
| bee_1 | 85 Hz | Banguela sp. | 0.31 | flight (0.92) |
| bee_2 | 66 Hz | *Bombus morio* | 0.33 | flight (0.88) |
| bee_3 | 64 Hz | Banguela sp. | 0.31 | flight (0.87) |
| bee_100 | 57 Hz | Banguela sp. | 0.32 | flight (0.88) |
| drone_1 | 78 Hz | Xylocopa pretinha | 0.30 | flight (0.61) |
| drone_2 | 84 Hz | Xylocopa pretinha | 0.37 | flight (0.67) |

Reading: *Apis* maps to nearby Apidae/Colletidae neighbors (*Bombus* is the same
family as *Apis*), never with high confidence (0.30–0.37). Drones (low hum) map
to Xylocopa, the largest low-frequency bee in the training set. Buzz type says
flight with high confidence, which is sensible for continuous hive hum (no
sonication pulses). Graceful degradation, not collapse.

## Probe 3 — negative control (Kaggle InsectSound1000, aphids)

2 × *Myzus persicae* greenhouse recordings (tiny, near-silent insects).
Result: *Bombus morio* at 0.37–0.49, buzz type floral at ~0.8.
The model answers confidently-ish on near-silence. This is the case for an
unknown-species / low-signal rejection threshold (open work, see README
limitations). A 2.5 s greenhouse background should not return a bee species
at 0.49 without a warning.

## Takeaway

- In-domain: works (5/5).
- Nearby out-of-distribution (other bees): nearest-neighbor behavior with low
  confidence — the honest signal to threshold on.
- Far out-of-distribution (near-silence): overconfident — needs a rejection
  rule before any field use.
