# Personal Garmin Dashboard (GitHub-style)

A dark-mode, GitHub-inspired personal fitness & health dashboard built with **Streamlit** and **Plotly**, powered by the **Official Garmin Connect Personal API** (`https://personal.connectapi.garmin.com`).

## Features
- **GitHub-style Contribution Heatmaps** for daily training minutes, running/cycling/walking distance, and nightly sleep hours
- **Overview**: Current & longest activity streaks, total volume, weekly sport breakdown, daily step trends
- **Running / Cycling / Walking**: Dedicated sport tabs with daily heatmaps, weekly mileage, pace/speed vs. heart-rate scatter plots, and session tables
- **Sleep**: Nightly sleep heatmap, stacked sleep stages (Deep, REM, Light, Awake), and sleep scores
- **Recovery & Fitness**: VO₂max trend, nightly HRV with personal balanced baseline band, resting heart rate, and Body Battery
- **Race Predictions**: 5K, 10K, Half Marathon, and Marathon times & paces
- **Activity Log**: Filterable feed of recent workouts

---

## 1. Get Your Official Garmin Personal API Key
1. Sign in to **Garmin Connect Web** (`https://connect.garmin.com`).
2. Go to **Account Settings → Personal API Key** (or visit the Garmin Connect Personal API portal).
3. Click **Create Key**, choose an expiration window (up to 1 year), and copy the generated key immediately (it is only shown once).

How authentication works under the hood:
- The app exchanges your `api_key` via `POST https://personal.connectapi.garmin.com/personalapitoken-service/personal-api/personal-api-key/token` (`Authorization: ApiKey <your-api-key>`) for a 24-hour OAuth2 `Bearer` token and automatically refreshes it when needed.

---

## 2. Run Locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit `.streamlit/secrets.toml` and paste your key:
```toml
[garmin]
api_key = "your-garmin-personal-api-key"
```
Then launch:
```bash
streamlit run app.py
```
*(You can also paste your Personal API Key directly into the sidebar or toggle **Demo data** to preview the dashboard without a key.)*

---

## 3. Deploy on Streamlit Community Cloud via GitHub
1. Push this repository to GitHub (`.gitignore` already excludes `.streamlit/secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, select your repo, and set **Main file path** to `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   [garmin]
   api_key = "your-garmin-personal-api-key"
   ```
4. Click **Deploy!**
