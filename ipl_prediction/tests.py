"""
Tests for the IPL Score Prediction application.
Run with: python -m pytest tests/ -v  (from ipl_prediction/)
"""

import sys
import os

# Ensure the ipl_prediction package is importable
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import json

from model_utils import (
    generate_dataset,
    build_pipeline,
    TEAMS,
    CITIES,
    predict_score,
)
from app import app as flask_app


# ---------------------------------------------------------------------------
# Dataset generation tests
# ---------------------------------------------------------------------------


def test_generate_dataset_shape():
    df = generate_dataset(n_matches=50)
    assert len(df) > 0, "Dataset should not be empty"
    expected_cols = {
        "batting_team", "bowling_team", "city",
        "overs", "runs", "wickets",
        "runs_last_5", "wickets_last_5", "total_runs",
    }
    assert expected_cols.issubset(set(df.columns))


def test_generate_dataset_value_ranges():
    df = generate_dataset(n_matches=100)
    assert df["overs"].between(1, 20).all(), "Overs out of range"
    assert (df["runs"] >= 0).all(), "Negative runs"
    assert df["wickets"].between(0, 10).all(), "Wickets out of range"
    assert (df["total_runs"] > 0).all(), "Final score must be positive"


def test_teams_and_cities_constants():
    assert len(TEAMS) == 10
    assert len(CITIES) == 10
    assert "Mumbai Indians" in TEAMS
    assert "Mumbai" in CITIES


# ---------------------------------------------------------------------------
# Model / pipeline tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_pipeline():
    df = generate_dataset(n_matches=200)
    X = df[["batting_team", "bowling_team", "city",
            "overs", "runs", "wickets", "runs_last_5", "wickets_last_5"]]
    y = df["total_runs"]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def test_pipeline_prediction_range(trained_pipeline):
    result = predict_score(
        trained_pipeline,
        batting_team="Mumbai Indians",
        bowling_team="Chennai Super Kings",
        city="Mumbai",
        overs=10,
        runs=80,
        wickets=2,
        runs_last_5=38,
        wickets_last_5=1,
    )
    assert 50 <= result["predicted_score"] <= 350, "Prediction out of realistic range"
    assert result["low"] <= result["predicted_score"] <= result["high"]


def test_pipeline_unknown_team_handled(trained_pipeline):
    """Pipeline should not raise when using handle_unknown='ignore'."""
    result = predict_score(
        trained_pipeline,
        batting_team="Unknown Team",
        bowling_team="Chennai Super Kings",
        city="Mumbai",
        overs=10,
        runs=80,
        wickets=2,
        runs_last_5=38,
        wickets_last_5=1,
    )
    assert isinstance(result["predicted_score"], int)


# ---------------------------------------------------------------------------
# Flask API tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(trained_pipeline):
    flask_app.config["TESTING"] = True
    # Inject the already-trained pipeline so the API doesn't re-train
    import app as app_module
    app_module._pipeline = trained_pipeline
    with flask_app.test_client() as c:
        yield c


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"IPL Score Predictor" in resp.data


def test_api_teams(client):
    resp = client.get("/api/teams")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, list)
    assert len(data) == 10


def test_api_cities(client):
    resp = client.get("/api/cities")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data) == 10


def test_api_predict_valid(client):
    payload = {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "city": "Mumbai",
        "overs": 10,
        "runs": 85,
        "wickets": 2,
        "runs_last_5": 40,
        "wickets_last_5": 1,
    }
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "predicted_score" in data
    assert "low" in data
    assert "high" in data
    assert data["low"] <= data["predicted_score"] <= data["high"]


def test_api_predict_same_teams(client):
    payload = {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Mumbai Indians",
        "city": "Mumbai",
        "overs": 10,
        "runs": 85,
        "wickets": 2,
        "runs_last_5": 40,
        "wickets_last_5": 1,
    }
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_api_predict_missing_field(client):
    payload = {
        "batting_team": "Mumbai Indians",
        # bowling_team missing
        "city": "Mumbai",
        "overs": 10,
        "runs": 85,
        "wickets": 2,
        "runs_last_5": 40,
        "wickets_last_5": 1,
    }
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_api_predict_invalid_overs(client):
    payload = {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "city": "Mumbai",
        "overs": 25,  # invalid – > 20
        "runs": 85,
        "wickets": 2,
        "runs_last_5": 40,
        "wickets_last_5": 1,
    }
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_api_predict_invalid_wickets(client):
    payload = {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "city": "Mumbai",
        "overs": 10,
        "runs": 85,
        "wickets": 11,  # invalid
        "runs_last_5": 40,
        "wickets_last_5": 1,
    }
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400
