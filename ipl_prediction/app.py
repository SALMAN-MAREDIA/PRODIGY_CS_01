"""
IPL Score Prediction – Flask Application
"""

import os
from flask import Flask, jsonify, render_template, request

from model_utils import (
    CITIES,
    TEAMS,
    load_model,
    predict_score,
    train_and_save_model,
    MODEL_PATH,
)

app = Flask(__name__)

# Load (or train) model at startup
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_model()
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", teams=TEAMS, cities=CITIES)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # -- Validate required fields --
    required = [
        "batting_team",
        "bowling_team",
        "city",
        "overs",
        "runs",
        "wickets",
        "runs_last_5",
        "wickets_last_5",
    ]
    missing = [f for f in required if f not in data or data[f] == ""]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    batting_team = str(data["batting_team"]).strip()
    bowling_team = str(data["bowling_team"]).strip()
    city = str(data["city"]).strip()

    if batting_team == bowling_team:
        return jsonify({"error": "Batting and bowling teams must be different."}), 400
    if batting_team not in TEAMS:
        return jsonify({"error": f"Unknown batting team: {batting_team}"}), 400
    if bowling_team not in TEAMS:
        return jsonify({"error": f"Unknown bowling team: {bowling_team}"}), 400
    if city not in CITIES:
        return jsonify({"error": f"Unknown city: {city}"}), 400

    try:
        overs = float(data["overs"])
        runs = int(data["runs"])
        wickets = int(data["wickets"])
        runs_last_5 = int(data["runs_last_5"])
        wickets_last_5 = int(data["wickets_last_5"])
    except (ValueError, TypeError):
        return jsonify({"error": "Numeric fields must be valid numbers."}), 400

    if not (1 <= overs <= 20):
        return jsonify({"error": "Overs must be between 1 and 20."}), 400
    if not (0 <= runs <= 400):
        return jsonify({"error": "Current runs must be between 0 and 400."}), 400
    if not (0 <= wickets <= 10):
        return jsonify({"error": "Wickets must be between 0 and 10."}), 400
    if not (0 <= runs_last_5 <= 150):
        return jsonify({"error": "Runs in last 5 overs must be between 0 and 150."}), 400
    if not (0 <= wickets_last_5 <= 10):
        return jsonify({"error": "Wickets in last 5 overs must be between 0 and 10."}), 400

    result = predict_score(
        get_pipeline(),
        batting_team=batting_team,
        bowling_team=bowling_team,
        city=city,
        overs=overs,
        runs=runs,
        wickets=wickets,
        runs_last_5=runs_last_5,
        wickets_last_5=wickets_last_5,
    )
    return jsonify(result)


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Retrain the model on a freshly generated dataset."""
    global _pipeline
    metrics = train_and_save_model(MODEL_PATH)
    _pipeline = load_model(MODEL_PATH)
    return jsonify({"message": "Model retrained successfully.", "metrics": metrics})


@app.route("/api/teams")
def teams():
    return jsonify(TEAMS)


@app.route("/api/cities")
def cities():
    return jsonify(CITIES)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure model exists before first request
    get_pipeline()
    app.run(debug=False, host="0.0.0.0", port=5000)
