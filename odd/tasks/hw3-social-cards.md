# Feature: publishable social cards for Homework 3 (03-classification)

## Objective
Produce ready-to-post image cards (LinkedIn, Instagram, X) that present the six
Homework 3 answers with their supporting metrics, plus matching post captions.

## Problem / Why
The user solved Homework 3 (ML Zoomcamp 2026, lead-scoring classification) and
wants publishable visuals instead of a screenshot of a notebook. Cards must be
built from the real computed numbers so nothing on them is invented.

## Scope
- One self-contained generator script: `homework-3/social/generate_social_cards.py`
- Bundled Inter fonts (SIL OFL) so rendering is reproducible offline
- Outputs: `homework-3/social/out/<platform>-*.png`
- Captions: `homework-3/social/captions.md`

## Authorized scope (allowed edit surfaces)
- `odd/tasks/hw3-social-cards.md`
- `homework-3/social/generate_social_cards.py`
- `homework-3/social/captions.md`
- `homework-3/social/fonts/`
- `homework-3/social/out/`

Out of scope: touching notebooks in the repo root, pushing, PR creation.

## Data of record (verified in homework-3.ipynb, executed)
- Q1 mode(industry) = technology (1173; NA=240 after fill)
- Q2 biggest correlation = interaction_count & lead_score = 0.9157
- Q3 biggest MI (train, round 2) = lead_source = 0.03
- Q4 accuracy = 0.645 -> 0.65
- Q5 smallest elimination diff = number_of_courses_viewed (+0.002)
- Q6 best C = 0.001 (0.645)
- Dataset: 5000 rows, converted 57.7% positive, split 4000/3000/1000

## Checklist

- [x] T1 - Design system: theme tokens (ink + gold), Inter weights, rounded-card and pill primitives, 2x supersampled rendering. Route: inline (single file, design resolved).
      Trigger evidence: no mapping trigger (all numbers already measured); no writer trigger (one source file; PNGs are outputs).
- [x] T2 - LinkedIn card 1200x1200 with all six answers + metric chips. Route: inline.
- [x] T3 - Instagram post 1080x1350 and story 1080x1920. Route: inline.
- [x] T4 - X card 1600x900 (two-column answer grid, larger type). Route: inline.
- [x] T5 - Verification: exact pixel dimensions, non-blank, no clipped text, Inter actually resolved. Route: inline.
- [x] T6 - Captions file for the three platforms (Spanish, technical terms in English). Route: inline.
- [x] T7 - Work-unit commit + native `gentle-ai review assess` (result: risk `passive`, reason `non_executable_only`).
- [x] T8 - Code-snippet series: 4 LinkedIn cards, Pygments highlighting, Fira Code, line numbers, result strip, dataset notes. Route: inline.
- [x] T9 - Carousel captions for the code series. Route: inline.
- [ ] T10 - Work-unit commit for the code series + `gentle-ai review assess --base-ref 93a1ef1`.

## Acceptance criteria
- Every number on a card matches the executed notebook output.
- Files land at the exact platform pixel sizes: 1200x1200, 1080x1350, 1080x1920, 1600x900.
- No default DejaVu Sans anywhere; all text renders in Inter.
- No text overflows its container and no element crosses the safe margin.

## Progress

- T1 done: design system in `social/generate_social_cards.py` (ink + gold theme, Inter 400-900 bundled in `social/fonts`, rounded/pill primitives, 2x supersampled output, auto `fit()` shrink + `check_bounds()` overflow guard).
- T2 done: `out/linkedin-post-1200x1200.png`.
- T3 done: `out/instagram-post-1080x1350.png`, `out/instagram-story-1080x1920.png`.
- T4 done: `out/x-card-1600x900.png` (2 columns x 3 rows).
- T5 done: all four files verified at exact platform pixel sizes, opaque RGB, zero text-overflow violations from the built-in guard.
- T6 done: `social/captions.md` with per-platform captions + alt text.
- T7 done: commit `93a1ef1` on `feat/hw3-social-cards` (captions + this doc).
  `gentle-ai review assess` on that candidate returned risk `passive`
  (`non_executable_only`), so no consent or reviewer ceremony applied.
  Note: the cards, generator, fonts and notebook were already committed by the
  user on `main` as `3b5f5b9` while this task was in flight; nothing was
  duplicated.
- T8 done: 4 code cards, `out/linkedin-code-0*.png`, 1200x1200.
- T9 done: carousel caption added to `social/captions.md`.

## Design decisions

- Answers are gold, structure is neutral grey: the card reads as an answer key where only the answers are "highlighted".
- Cards stay typographic (no charts) so they remain legible at thumbnail size.
- Header rule and answer rows use hairlines instead of boxes to keep the ink surface calm.
- Output is flattened to opaque RGB because social uploaders handle alpha inconsistently.
- Code cards use Pygments for real token classification and bundled Fira Code (SIL OFL), so highlighting is not hand-rolled. Pygments is optional: without it the code renders in a single colour instead of failing.
- Code font size is derived from the longest line, and short snippets get
  airier leading instead of a dead band at the bottom of the card.
- Snippets are abbreviated for legibility (`train`/`val` instead of
  `df_train`/`df_val`, calls split over two lines). The captions say so
  explicitly, and the full executed code lives in `homework-3.ipynb`.
