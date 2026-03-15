"""
IPL Score Prediction – Dataset generation and model training utilities.

Generates a synthetic-but-realistic IPL dataset and trains a
RandomForestRegressor to predict the final innings score given the
current match state.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

CITIES = [
    "Mumbai",
    "Chennai",
    "Bengaluru",
    "Kolkata",
    "Delhi",
    "Mohali",
    "Jaipur",
    "Hyderabad",
    "Ahmedabad",
    "Lucknow",
]

# Relative batting strengths (used for data generation only)
TEAM_BATTING_STRENGTH = {
    "Mumbai Indians": 1.05,
    "Chennai Super Kings": 1.03,
    "Royal Challengers Bengaluru": 1.08,
    "Kolkata Knight Riders": 1.00,
    "Delhi Capitals": 0.98,
    "Punjab Kings": 1.04,
    "Rajasthan Royals": 1.01,
    "Sunrisers Hyderabad": 0.97,
    "Gujarat Titans": 1.02,
    "Lucknow Super Giants": 0.99,
}

# Home-ground / pitch advantage
CITY_FACTOR = {
    "Mumbai": 1.02,
    "Chennai": 0.97,
    "Bengaluru": 1.05,
    "Kolkata": 1.00,
    "Delhi": 1.01,
    "Mohali": 1.03,
    "Jaipur": 1.00,
    "Hyderabad": 0.98,
    "Ahmedabad": 0.99,
    "Lucknow": 1.01,
}

FEATURES = [
    "batting_team",
    "bowling_team",
    "city",
    "overs",
    "runs",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ipl_model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "ipl_data.csv")

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _simulate_innings(batting_team: str, city: str, rng: np.random.Generator) -> dict:
    """
    Simulate a T20 innings ball-by-ball and return over-level statistics.

    Returns a dict with lists: runs_after_over, wickets_after_over
    """
    strength = TEAM_BATTING_STRENGTH[batting_team]
    city_f = CITY_FACTOR[city]
    base_rr = rng.normal(8.0 * strength * city_f, 0.8)  # target run-rate

    # Phase multipliers: powerplay (1-6), middle (7-15), death (16-20)
    phase_mult = (
        [rng.normal(0.85, 0.05)] * 6
        + [rng.normal(1.0, 0.05)] * 9
        + [rng.normal(1.15, 0.08)] * 5
    )

    runs_after_over = []
    wickets_after_over = []
    total_runs = 0
    total_wkts = 0

    for ov in range(20):
        if total_wkts >= 10:
            # All out — pad remaining overs
            runs_after_over.append(total_runs)
            wickets_after_over.append(total_wkts)
            continue

        remaining_wkts = 10 - total_wkts
        wkt_prob = max(0.04, 0.15 - remaining_wkts * 0.008)
        wkts_this_over = int(rng.binomial(4, wkt_prob))
        wkts_this_over = min(wkts_this_over, remaining_wkts)

        rr_this_over = max(0, rng.normal(base_rr * phase_mult[ov], 1.5))
        # Reduce run-rate if wickets fall
        rr_this_over = max(0, rr_this_over - wkts_this_over * rng.uniform(1, 2))

        over_runs = int(round(rr_this_over))
        total_runs += over_runs
        total_wkts += wkts_this_over

        runs_after_over.append(total_runs)
        wickets_after_over.append(total_wkts)

    return {"runs_after_over": runs_after_over, "wickets_after_over": wickets_after_over}


def generate_dataset(n_matches: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic IPL dataset and return a DataFrame."""
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n_matches):
        batting_team = rng.choice(TEAMS)
        bowling_team = rng.choice([t for t in TEAMS if t != batting_team])
        city = rng.choice(CITIES)

        innings = _simulate_innings(batting_team, city, rng)
        runs_ao = innings["runs_after_over"]
        wkts_ao = innings["wickets_after_over"]
        final_score = runs_ao[19]

        # Create snapshots at overs 5, 10, 15 (enough balls have been bowled
        # to make a meaningful prediction; skip overs with 10 wickets already)
        for ov_idx in (4, 9, 14):
            if wkts_ao[ov_idx] >= 10:
                continue
            current_over = ov_idx + 1  # 1-indexed
            current_runs = runs_ao[ov_idx]
            current_wkts = wkts_ao[ov_idx]

            # Runs and wickets in the last 5 overs
            start = max(0, ov_idx - 4)
            runs_last_5 = current_runs - (runs_ao[start - 1] if start > 0 else 0)
            wkts_last_5 = current_wkts - (wkts_ao[start - 1] if start > 0 else 0)

            records.append(
                {
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "city": city,
                    "overs": current_over,
                    "runs": current_runs,
                    "wickets": current_wkts,
                    "runs_last_5": runs_last_5,
                    "wickets_last_5": wkts_last_5,
                    "total_runs": final_score,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessing and RF regressor."""
    categorical = ["batting_team", "bowling_team", "city"]
    numerical = ["overs", "runs", "wickets", "runs_last_5", "wickets_last_5"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_and_save_model(save_path: str = MODEL_PATH) -> dict:
    """
    Generate dataset, train the model, save it to *save_path*, and
    return evaluation metrics.
    """
    print("Generating IPL dataset …")
    df = generate_dataset(n_matches=3000)

    X = df[FEATURES]
    y = df["total_runs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}  |  R²: {r2:.4f}")

    joblib.dump(pipeline, save_path)
    print(f"Model saved → {save_path}")

    # Save dataset for reference
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset saved → {DATASET_PATH}")

    return {"mae": round(mae, 2), "r2": round(r2, 4), "samples": len(df)}


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    """Load the saved model; train first if it doesn't exist."""
    if not os.path.exists(model_path):
        train_and_save_model(model_path)
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_score(
    pipeline: Pipeline,
    batting_team: str,
    bowling_team: str,
    city: str,
    overs: float,
    runs: int,
    wickets: int,
    runs_last_5: int,
    wickets_last_5: int,
) -> dict:
    """
    Return the predicted final score with a ±confidence interval.
    """
    X = pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "overs": overs,
                "runs": runs,
                "wickets": wickets,
                "runs_last_5": runs_last_5,
                "wickets_last_5": wickets_last_5,
            }
        ]
    )

    prediction = pipeline.predict(X)[0]
    # Estimated confidence band (roughly ±8% of prediction)
    margin = prediction * 0.08
    return {
        "predicted_score": int(round(prediction)),
        "low": int(round(max(runs, prediction - margin))),
        "high": int(round(prediction + margin)),
    }
