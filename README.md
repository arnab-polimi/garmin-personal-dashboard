# Personal Garmin & Hevy Dashboard (GitHub-style)

A dark-mode, GitHub-inspired personal fitness, weightlifting & health dashboard built with **Streamlit** and **Plotly**, powered by the **Official Garmin Connect Personal API** (`https://personal.connectapi.garmin.com`) and the **Official Hevy API v1** (`https://api.hevyapp.com/v1`).

## Features
- **GitHub-style Contribution Heatmaps** for daily training minutes, running/cycling/walking distance, weightlifting volume (kg), and nightly sleep hours
- **Overview**: Current & longest activity streaks, total volume, weekly sport breakdown, daily step trends
- **Running / Cycling / Walking**: Dedicated sport tabs with daily heatmaps, weekly mileage, pace/speed vs. heart-rate scatter plots, and session tables
- **Weightlifting (Hevy)**: Total tonnage, working sets & reps, weekly volume, muscle group split, interactive exercise progression & Epley 1RM tracker, Personal Records leaderboard, saved Hevy routines, and expandable set-by-set workout logs
- **Sleep**: Nightly sleep heatmap, stacked sleep stages (Deep, REM, Light, Awake), and sleep scores
- **Recovery & Fitness**: VO₂max trend, nightly HRV with personal balanced baseline band, resting heart rate, and Body Battery
- **Race Predictions**: 5K, 10K, Half Marathon, and Marathon times & paces
- **Activity Log**: Filterable feed of recent workouts

---

## 1. Get Your API Keys
### Garmin Personal API Key
1. Sign in to **Garmin Connect Web** (`https://connect.garmin.com`).
2. Go to **Account Settings → Personal API Key** (or visit the Garmin Connect Personal API portal).
3. Click **Create Key**, choose an expiration window (up to 1 year), and copy the generated key immediately.

### Hevy API Key (Weightlifting)
1. Sign in to **Hevy Web** and open **[hevy.com/settings?developer](https://hevy.com/settings?developer)**.
2. Copy your Hevy Pro developer API key (`UUID` format, sent via `api-key` header to `https://api.hevyapp.com/v1`).

---

## 2. Run Locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit `.streamlit/secrets.toml` and paste your keys:
```toml
[garmin]
api_key = "your-garmin-personal-api-key"

[hevy]
api_key = "your-hevy-api-key-uuid"
```
Then launch:
```bash
streamlit run app.py
```
*(You can also paste your Garmin or Hevy API keys directly into the sidebar or toggle **Demo data** to preview the dashboard without keys.)*

---

## 3. Deploy on Streamlit Community Cloud & Keep Awake 24/7
1. Push this repository to GitHub (`.gitignore` already excludes `.streamlit/secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, select your repo, and set **Main file path** to `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   [garmin]
   api_key = "your-garmin-personal-api-key"

   [hevy]
   api_key = "your-hevy-api-key-uuid"
   ```
4. **Prevent Daily Hibernation (GitHub Action)**:
   - Included in `.github/workflows/keep-alive.yml` is a scheduled workflow (runs every 6 hours) that visits your Streamlit app with headless Chromium (Playwright), establishes a real WebSocket session, and automatically clicks *"Yes, get this app back up!"* if it ever sleeps.
   - In your GitHub repo, go to **Settings → Secrets and variables → Actions** and add a Repository Variable or Secret named `STREAMLIT_APP_URL` set to your deployed URL (e.g., `https://your-app-name.streamlit.app`).
