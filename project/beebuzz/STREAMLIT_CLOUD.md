# BeeBuzz on Streamlit Community Cloud
# https://share.streamlit.io
#
# 1. Push this repo to GitHub (public).
# 2. Go to https://share.streamlit.io and sign in with GitHub.
# 3. "New app" → select this repo → main file: dashboard/app.py
# 4. Deploy. The app builds from requirements.txt automatically.
#
# Notes:
# - models/*.joblib are committed, so no training happens at deploy time.
# - The dashboard reads data/raw/buzz_data.csv at runtime for the example CSV.
