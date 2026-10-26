# Homework 5 — Answers

## Q1 — environment check
- `uv --version` → `uv 0.12.12`

## Q2 — locked dependency
- scikit-learn version in uv.lock/pyproject.toml → `1.7.2`

## Q3 — load the model
- lead 1 conversion_probability → `0.533`

## Q4 — serve the model
- lead 2 conversion_probability → `0.770`

## Q5 — container configuration
- Python base-image tag in Dockerfile → `python:3.11.15-slim-bookworm`

## Q6 — run the container
- lead 2 conversion_probability (container) → `0.770`
- health → `{"status":"ok", ...}` with matching model_sha256
