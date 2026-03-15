# PRODIGY_CS_01

This repository contains two projects:

---

## 1. 🔐 Caesar Cipher (task1.py)

A simple Caesar Cipher encryption/decryption tool in Python.

**Run:**
```bash
python task1.py
```

---

## 2. 🏏 IPL Score Predictor (ipl_prediction/)

A full-stack web application that predicts the final IPL innings score given
the current match state, powered by a **Random Forest** machine-learning model.

### Features
- **Machine Learning** – Random Forest Regressor trained on synthetic-realistic
  IPL innings data (3,000 matches × 3 over-snapshots each).
- **REST API** – Flask backend with `/api/predict` endpoint.
- **Responsive UI** – Bootstrap 5 frontend with live form validation, confidence
  band display, and match statistics.
- **Input signals** – Batting team, bowling team, venue city, overs completed,
  current score, wickets fallen, runs / wickets in the last 5 overs.
- **Output** – Predicted final score with a low/high confidence range.

### Tech Stack
| Layer      | Technology                           |
|------------|--------------------------------------|
| Backend    | Python 3, Flask 3                    |
| ML         | scikit-learn (Random Forest), joblib |
| Data       | pandas, numpy                        |
| Frontend   | HTML5, Bootstrap 5, Vanilla JS       |

### Getting Started

```bash
cd ipl_prediction

# Install dependencies
pip install -r requirements.txt

# Run the app (model is trained automatically on first launch)
python app.py
```

Then open **http://localhost:5000** in your browser.

### API Reference

#### `POST /api/predict`

**Request body (JSON):**

| Field            | Type   | Description                         |
|------------------|--------|-------------------------------------|
| `batting_team`   | string | Name of the batting team            |
| `bowling_team`   | string | Name of the bowling team            |
| `city`           | string | Venue city                          |
| `overs`          | float  | Overs completed (1–20)              |
| `runs`           | int    | Current score (0–400)               |
| `wickets`        | int    | Wickets fallen (0–10)               |
| `runs_last_5`    | int    | Runs scored in the last 5 overs     |
| `wickets_last_5` | int    | Wickets fallen in the last 5 overs  |

**Response (JSON):**

```json
{
  "predicted_score": 179,
  "low": 165,
  "high": 193
}
```

#### `POST /api/retrain`

Retrains the model on a freshly generated dataset and returns evaluation metrics.

---

## Domain
Cyber Security / Full-Stack ML – Prodigy InfoTech Internship
